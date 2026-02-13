import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import os
import SplatIO
import simd
import zlib
import Accelerate

#if arch(x86_64)
typealias Float16 = Float
#warning("x86_64 targets are unsupported by MetalSplatter and will fail at runtime. MetalSplatter builds on x86_64 only because Xcode builds Swift Packages as universal binaries and provides no way to override this. When Swift supports Float16 on x86_64, this may be revisited.")
#endif

public struct SPZColorSettings {
    public static var brightness: Float = 4.0    // More aggressive brightness boost
    public static var gamma: Float = 4.0         // Lower gamma for more contrast
}

public class SplatRenderer {
    
    /// Rendering mode for handling different splat formats
    public enum RenderingMode {
        case auto                    // Automatically detect from data
        case forceBasicColor        // Force basic color even if SH data exists
        case forceSphericalHarmonics // Force SH rendering (error if not available)
    }
    
    /// Current rendering mode
    public var renderingMode: RenderingMode = .auto
    
    /// Whether current loaded data supports spherical harmonics
    private var dataSupportsSphericalHarmonics: Bool = false
    
    /// Whether to use SH shaders based on current data and settings
    private var useSHRendering: Bool {
        switch renderingMode {
        case .auto:
            return dataSupportsSphericalHarmonics
        case .forceBasicColor:
            return false
        case .forceSphericalHarmonics:
            assert(dataSupportsSphericalHarmonics, "SH rendering forced but no SH data available")
            return dataSupportsSphericalHarmonics
        }
    }
    enum Constants {
        // Keep in sync with Shaders.metal : maxViewCount
        static let maxViewCount = 2
        // Sort by euclidian distance squared from camera position (true), or along the "forward" vector (false)
        // TODO: compare the behaviour and performance of sortByDistance
        // notes: sortByDistance introduces unstable artifacts when you get close to an object; whereas !sortByDistance introduces artifacts are you turn -- but they're a little subtler maybe?
        static let sortByDistance = true
        // Only store indices for 1024 splats; for the remainder, use instancing of these existing indices.
        // Setting to 1 uses only instancing (with a significant performance penalty); setting to a number higher than the splat count
        // uses only indexing (with a significant memory penalty for th elarge index array, and a small performance penalty
        // because that can't be cached as easiliy). Anywhere within an order of magnitude (or more?) of 1k seems to be the sweet spot,
        // with effectively no memory penalty compated to instancing, and slightly better performance than even using all indexing.
        static let maxIndexedSplatCount = 1024

        static let tileSize = MTLSize(width: 32, height: 32, depth: 1)
    }

    private static let log =
        Logger(subsystem: Bundle.module.bundleIdentifier!,
               category: "SplatRenderer")

    public struct ViewportDescriptor {
        public var viewport: MTLViewport
        public var projectionMatrix: simd_float4x4
        public var viewMatrix: simd_float4x4
        public var screenSize: SIMD2<Int>

        public init(viewport: MTLViewport, projectionMatrix: simd_float4x4, viewMatrix: simd_float4x4, screenSize: SIMD2<Int>) {
            self.viewport = viewport
            self.projectionMatrix = projectionMatrix
            self.viewMatrix = viewMatrix
            self.screenSize = screenSize
        }
    }

    // Keep in sync with Shaders.metal : BufferIndex
    enum BufferIndex: NSInteger {
        case uniforms = 0
        case splat    = 1
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels

        var splatCount: UInt32
        var indexedSplatCount: UInt32
    }

    // Keep in sync with Shaders.metal : UniformsArray
    struct UniformsArray {
        // maxViewCount = 2, so we have 2 entries
        var uniforms0: Uniforms
        var uniforms1: Uniforms

        // The 256 byte aligned size of our uniform structure
        static var alignedSize: Int { (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100 }

        mutating func setUniforms(index: Int, _ uniforms: Uniforms) {
            switch index {
            case 0: uniforms0 = uniforms
            case 1: uniforms1 = uniforms
            default: break
            }
        }
    }

    struct PackedHalf3 {
        var x: Float16
        var y: Float16
        var z: Float16
    }

    struct PackedRGBHalf4 {
        var r: Float16
        var g: Float16
        var b: Float16
        var a: Float16
    }

    // Keep in sync with Shaders.metal : Splat
    struct Splat {
        var position: MTLPackedFloat3
        var color: PackedRGBHalf4
        var covA: PackedHalf3
        var covB: PackedHalf3
    }

    struct SplatIndexAndDepth {
        var index: UInt32
        var depth: Float
    }

    public let device: MTLDevice
    public let colorFormat: MTLPixelFormat
    public let depthFormat: MTLPixelFormat
    public let sampleCount: Int
    public let maxViewCount: Int
    public let maxSimultaneousRenders: Int

    /**
     High-quality depth takes longer, but results in a continuous, more-representative depth buffer result, which is useful for reducing artifacts during Vision Pro's frame reprojection.
     */
    public var highQualityDepth: Bool = true

    private var writeDepth: Bool {
        depthFormat != .invalid
    }

    /**
     The SplatRenderer has two shader pipelines.
     - The single stage has a vertex shader, and a fragment shader. It can produce depth (or not), but the depth it produces is the depth of the nearest splat, whether it's visible or now.
     - The multi-stage pipeline uses a set of shaders which communicate using imageblock tile memory: initialization (which clears the tile memory), draw splats (similar to the single-stage
     pipeline but the end result is tile memory, not color+depth), and a post-process stage which merely copies the tile memory (color and optionally depth) to the frame's buffers.
     This is neccessary so that the primary stage can do its own blending -- of both color and depth -- by reading the previous values and writing new ones, which isn't possible without tile
     memory. Color blending works the same as the hardcoded path, but depth blending uses color alpha and results in mostly-transparent splats contributing only slightly to the depth,
     resulting in a much more continuous and representative depth value, which is important for reprojection on Vision Pro.
     */
    private var useMultiStagePipeline: Bool {
        // Multi-stage pipeline's initializeFragmentStore clears imageblock to black,
        // which overwrites camera background in AR mode. Use single-stage when preserving content.
        if preserveExistingContent {
            return false
        }
        
#if targetEnvironment(simulator)
        return false
#else
        return writeDepth && highQualityDepth
#endif
    }

    public var clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
    public var preserveExistingContent = false  // If true, uses .load instead of .clear
    

    public var onSortStart: (() -> Void)?
    public var onSortComplete: ((TimeInterval) -> Void)?

    private let library: MTLLibrary
    // Single-stage pipeline
    private var singleStagePipelineState: MTLRenderPipelineState?
    private var singleStageDepthState: MTLDepthStencilState?
    // Multi-stage pipeline
    private var initializePipelineState: MTLRenderPipelineState?
    private var drawSplatPipelineState: MTLRenderPipelineState?
    private var drawSplatDepthState: MTLDepthStencilState?
    private var postprocessPipelineState: MTLRenderPipelineState?
    private var postprocessDepthState: MTLDepthStencilState?

    // dynamicUniformBuffers contains maxSimultaneousRenders uniforms buffers,
    // which we round-robin through, one per render; this is managed by switchToNextDynamicBuffer.
    // uniforms = the i'th buffer (where i = uniformBufferIndex, which varies from 0 to maxSimultaneousRenders-1)
    var dynamicUniformBuffers: MTLBuffer
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>

    // cameraWorldPosition and Forward vectors are the latest mean camera position across all viewports
    var cameraWorldPosition: SIMD3<Float> = .zero
    var cameraWorldForward: SIMD3<Float> = .init(x: 0, y: 0, z: -1)

    typealias IndexType = UInt32
    // splatBuffer contains one entry for each gaussian splat
    var splatBuffer: MetalBuffer<Splat>
    // splatBufferPrime is a copy of splatBuffer, which is not currenly in use for rendering.
    // We use this for sorting, and when we're done, swap it with splatBuffer.
    // There's a good chance that we'll sometimes end up sorting a splatBuffer still in use for
    // rendering.
    // TODO: Replace this with a more robust multiple-buffer scheme to guarantee we're never actively sorting a buffer still in use for rendering
    var splatBufferPrime: MetalBuffer<Splat>

    var indexBuffer: MetalBuffer<UInt32>

    public var splatCount: Int { splatBuffer.count }

    var sorting = false
    var orderAndDepthTempSort: [SplatIndexAndDepth] = []

    // ========================================================================
    // SORTING CONFIGURATION
    // ========================================================================

    /// Available sorting algorithms
    public enum SortingAlgorithm: String, CaseIterable {
        case cpuStandard = "CPU Standard Sort"      // Swift's built-in sort
        case gpuHybrid = "GPU Hybrid"               // GPU depth + CPU sort + GPU reorder
        case gpuBitonic = "GPU Bitonic Sort"        // Full GPU bitonic sort
        case gpuRadix = "GPU Radix Sort"            // Full GPU radix sort (experimental)
    }

    /// Current sorting algorithm (can be changed at runtime)
    public var sortingAlgorithm: SortingAlgorithm = .gpuHybrid

    /// Benchmark results storage
    public struct SortBenchmarkResult {
        public let algorithm: SortingAlgorithm
        public let splatCount: Int
        public let depthComputeTime: TimeInterval    // Time to compute depths
        public let sortTime: TimeInterval            // Time for actual sorting
        public let reorderTime: TimeInterval         // Time to reorder splats
        public let totalTime: TimeInterval           // Total time

        public var description: String {
            String(format: "%@ (%d splats): depth=%.2fms, sort=%.2fms, reorder=%.2fms, total=%.2fms",
                   algorithm.rawValue, splatCount,
                   depthComputeTime * 1000, sortTime * 1000,
                   reorderTime * 1000, totalTime * 1000)
        }
    }

    /// Last benchmark result (updated after each sort if benchmarking is enabled)
    public private(set) var lastBenchmarkResult: SortBenchmarkResult?

    /// Enable detailed timing for benchmarking (slight performance overhead)
    public var enableSortBenchmarking = false

    // GPU Sorting resources
    public var useGPUSorting = true  // Enable GPU-based sorting by default
    private var depthBuffer: MTLBuffer?
    private var indexBufferForSort: MTLBuffer?
    private var sortedIndexBuffer: MTLBuffer?
    private var depthComputePipeline: MTLComputePipelineState?
    private var reorderPipeline: MTLComputePipelineState?
    private var bitonicSortPipeline: MTLComputePipelineState?
    private var bitonicSortLocalPipeline: MTLComputePipelineState?
    private var commandQueue: MTLCommandQueue?
    private var lastSortedSplatCount: Int = 0

    // Radix sort reusable buffers (avoid allocation every frame)
    private var radixKeys: [UInt32] = []
    private var radixVals: [UInt32] = []
    private var radixKeysTemp: [UInt32] = []
    private var radixValsTemp: [UInt32] = []

    // Uniforms for depth computation kernel
    struct DepthComputeUniforms {
        var cameraPosition: SIMD3<Float>
        var cameraForward: SIMD3<Float>
        var splatCount: UInt32
        var sortByDistance: Bool
    }

    // Uniforms for sorting kernels
    struct SortUniforms {
        var count: UInt32
        var stageDistance: UInt32
        var passDistance: UInt32
        var ascending: UInt32
    }

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
#if arch(x86_64)
        fatalError("MetalSplatter is unsupported on Intel architecture (x86_64)")
#endif

        self.device = device

        self.colorFormat = colorFormat
        self.depthFormat = depthFormat
        self.sampleCount = sampleCount
        self.maxViewCount = min(maxViewCount, Constants.maxViewCount)
        self.maxSimultaneousRenders = maxSimultaneousRenders

        let dynamicUniformBuffersSize = UniformsArray.alignedSize * maxSimultaneousRenders
        self.dynamicUniformBuffers = device.makeBuffer(length: dynamicUniformBuffersSize,
                                                       options: .storageModeShared)!
        self.dynamicUniformBuffers.label = "Uniform Buffers"
        self.uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents()).bindMemory(to: UniformsArray.self, capacity: 1)

        self.splatBuffer = try MetalBuffer(device: device)
        self.splatBufferPrime = try MetalBuffer(device: device)
        self.indexBuffer = try MetalBuffer(device: device)

        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            fatalError("Unable to initialize SplatRenderer: \(error)")
        }

        // Initialize GPU sorting infrastructure
        self.commandQueue = device.makeCommandQueue()
        setupGPUSortingPipelines()
    }

    private func setupGPUSortingPipelines() {
        do {
            // Create compute pipeline for depth computation
            if let depthFunction = library.makeFunction(name: "computeSplatDepthsDescending") {
                depthComputePipeline = try device.makeComputePipelineState(function: depthFunction)
            }
            // Create compute pipeline for reordering splats
            if let reorderFunction = library.makeFunction(name: "reorderSplats") {
                reorderPipeline = try device.makeComputePipelineState(function: reorderFunction)
            }
            // Create compute pipeline for bitonic sort
            if let bitonicFunction = library.makeFunction(name: "bitonicSortPass") {
                bitonicSortPipeline = try device.makeComputePipelineState(function: bitonicFunction)
            }
            // Create compute pipeline for local bitonic sort
            if let bitonicLocalFunction = library.makeFunction(name: "bitonicSortLocal") {
                bitonicSortLocalPipeline = try device.makeComputePipelineState(function: bitonicLocalFunction)
            }
        } catch {
            Self.log.error("Failed to create GPU sorting pipelines: \(error.localizedDescription)")
            useGPUSorting = false
        }
    }

    /// Returns the next power of 2 >= n
    private func nextPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }

    private func ensureGPUSortBuffers(splatCount: Int) {
        guard splatCount > 0 else { return }

        // For bitonic sort, we need power-of-2 sized buffers
        let paddedCount = nextPowerOf2(splatCount)
        let depthBufferSize = MemoryLayout<Float>.stride * paddedCount
        let indexBufferSize = MemoryLayout<UInt32>.stride * paddedCount

        // Reallocate if needed
        if depthBuffer == nil || depthBuffer!.length < depthBufferSize {
            depthBuffer = device.makeBuffer(length: depthBufferSize, options: .storageModeShared)
            depthBuffer?.label = "Depth Buffer for Sort"
        }
        if indexBufferForSort == nil || indexBufferForSort!.length < indexBufferSize {
            indexBufferForSort = device.makeBuffer(length: indexBufferSize, options: .storageModeShared)
            indexBufferForSort?.label = "Index Buffer for Sort (Input)"
        }
        if sortedIndexBuffer == nil || sortedIndexBuffer!.length < indexBufferSize {
            sortedIndexBuffer = device.makeBuffer(length: indexBufferSize, options: .storageModeShared)
            sortedIndexBuffer?.label = "Index Buffer for Sort (Output)"
        }

        lastSortedSplatCount = splatCount
    }

    // MARK: - Radix Sort Implementation (O(n) instead of O(n log n))

    /// Radix sort for float keys with associated indices
    /// Uses 4 passes (8 bits each) for O(4n) = O(n) complexity
    /// Optimized with unsafe pointers to eliminate bounds checking overhead
    private func radixSortFloatIndices(
        depths: UnsafeMutablePointer<Float>,
        indices: UnsafeMutablePointer<UInt32>,
        output: UnsafeMutablePointer<UInt32>,
        count: Int
    ) {
        guard count > 0 else { return }

        // Ensure reusable buffers are large enough
        if radixKeys.count < count {
            radixKeys = [UInt32](repeating: 0, count: count)
            radixVals = [UInt32](repeating: 0, count: count)
            radixKeysTemp = [UInt32](repeating: 0, count: count)
            radixValsTemp = [UInt32](repeating: 0, count: count)
        }

        // Use withUnsafeMutableBufferPointer to eliminate bounds checking
        radixKeys.withUnsafeMutableBufferPointer { keysBuf in
            radixVals.withUnsafeMutableBufferPointer { valsBuf in
                radixKeysTemp.withUnsafeMutableBufferPointer { keysTempBuf in
                    radixValsTemp.withUnsafeMutableBufferPointer { valsTempBuf in
                        var keysPtr = keysBuf.baseAddress!
                        var valsPtr = valsBuf.baseAddress!
                        var keysTempPtr = keysTempBuf.baseAddress!
                        var valsTempPtr = valsTempBuf.baseAddress!

                        // Convert floats to sortable uint32 keys (IEEE 754 trick)
                        for i in 0..<count {
                            let floatBits = depths[i].bitPattern
                            if floatBits & 0x80000000 != 0 {
                                keysPtr[i] = ~floatBits  // Negative: flip all bits
                            } else {
                                keysPtr[i] = floatBits ^ 0x80000000  // Positive: flip sign bit
                            }
                            valsPtr[i] = indices[i]
                        }

                        // 4 passes with 8-bit radix (256 buckets)
                        var histogram = [Int](repeating: 0, count: 256)
                        let radixMask: UInt32 = 255

                        for pass in 0..<4 {
                            let shift = pass * 8

                            // Reset histogram using memset for speed
                            histogram.withUnsafeMutableBufferPointer { histPtr in
                                memset(histPtr.baseAddress!, 0, 256 * MemoryLayout<Int>.stride)
                            }

                            // Count histogram - use pointer arithmetic
                            histogram.withUnsafeMutableBufferPointer { histPtr in
                                let hPtr = histPtr.baseAddress!
                                for i in 0..<count {
                                    let digit = Int((keysPtr[i] >> shift) & radixMask)
                                    hPtr[digit] += 1
                                }

                                // Prefix sum (exclusive scan)
                                var sum = 0
                                for i in 0..<256 {
                                    let temp = hPtr[i]
                                    hPtr[i] = sum
                                    sum += temp
                                }

                                // Scatter to temp buffers
                                for i in 0..<count {
                                    let digit = Int((keysPtr[i] >> shift) & radixMask)
                                    let destIdx = hPtr[digit]
                                    keysTempPtr[destIdx] = keysPtr[i]
                                    valsTempPtr[destIdx] = valsPtr[i]
                                    hPtr[digit] += 1
                                }
                            }

                            // Swap pointers (no data movement!)
                            swap(&keysPtr, &keysTempPtr)
                            swap(&valsPtr, &valsTempPtr)
                        }

                        // Copy sorted indices to output
                        memcpy(output, valsPtr, count * MemoryLayout<UInt32>.stride)
                    }
                }
            }
        }
    }

    public func reset() {
        splatBuffer.count = 0
        try? splatBuffer.setCapacity(0)
    }

    public func read(from url: URL) async throws {
        print("🔄 SplatRenderer.read() called for: \(url.lastPathComponent)")
        
        let fileExtension = url.pathExtension.lowercased()
        
        if fileExtension == "spz" {
            // Handle SPZ files
            let points = try await SPZSceneReader.read(from: url)
            try add(points)
        } else {
            // Handle PLY/SPLAT files (existing code)
            var newPoints = SplatMemoryBuffer()
            try await newPoints.read(from: try AutodetectSceneReader(url))
            try add(newPoints.points)
        }
    }

    private func resetPipelineStates() {
        singleStagePipelineState = nil
        initializePipelineState = nil
        drawSplatPipelineState = nil
        drawSplatDepthState = nil
        postprocessPipelineState = nil
        postprocessDepthState = nil
    }

    private func buildSingleStagePipelineStatesIfNeeded() throws {
        guard singleStagePipelineState == nil else { return }

        singleStagePipelineState = try buildSingleStagePipelineState()
        singleStageDepthState = try buildSingleStageDepthState()
    }

    private func buildMultiStagePipelineStatesIfNeeded() throws {
        guard initializePipelineState == nil else { return }

        initializePipelineState = try buildInitializePipelineState()
        drawSplatPipelineState = try buildDrawSplatPipelineState()
        drawSplatDepthState = try buildDrawSplatDepthState()
        postprocessPipelineState = try buildPostprocessPipelineState()
        postprocessDepthState = try buildPostprocessDepthState()
    }

    private func buildSingleStagePipelineState() throws -> MTLRenderPipelineState {
        assert(!useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "SingleStagePipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "singleStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeRequiredFunction(name: "singleStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.pixelFormat = colorFormat
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .one
        colorAttachment.sourceAlphaBlendFactor = .one
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0] = colorAttachment

        // Only set depth format if we're actually using depth buffer  
        // In AR mode with preserveExistingContent=true, depthTexture is nil
        pipelineDescriptor.depthAttachmentPixelFormat = preserveExistingContent ? .invalid : depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildSingleStageDepthState() throws -> MTLDepthStencilState {
        assert(!useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        // Only enable depth writes if we have depth buffer AND not preserving existing content
        depthStateDescriptor.isDepthWriteEnabled = writeDepth && !preserveExistingContent
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    private func buildInitializePipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = library.makeRequiredFunction(name: "initializeFragmentStore")
        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

        return try device.makeRenderPipelineState(tileDescriptor: pipelineDescriptor, options: [], reflection: nil)
    }

    private func buildDrawSplatPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DrawSplatPipeline"
        pipelineDescriptor.vertexFunction = library.makeRequiredFunction(name: "multiStageSplatVertexShader")
        pipelineDescriptor.fragmentFunction = library.makeRequiredFunction(name: "multiStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDrawSplatDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        // Only enable depth writes if we have depth buffer AND not preserving existing content
        depthStateDescriptor.isDepthWriteEnabled = writeDepth && !preserveExistingContent
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    private func buildPostprocessPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "PostprocessPipeline"
        pipelineDescriptor.vertexFunction =
            library.makeRequiredFunction(name: "postprocessVertexShader")
        pipelineDescriptor.fragmentFunction =
            writeDepth
            ? library.makeRequiredFunction(name: "postprocessFragmentShader")
            : library.makeRequiredFunction(name: "postprocessFragmentShaderNoDepth")

        pipelineDescriptor.colorAttachments[0]!.pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildPostprocessDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        // Only enable depth writes if we have depth buffer AND not preserving existing content
        depthStateDescriptor.isDepthWriteEnabled = writeDepth && !preserveExistingContent
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    public func ensureAdditionalCapacity(_ pointCount: Int) throws {
        try splatBuffer.ensureCapacity(splatBuffer.count + pointCount)
    }

    public func add(_ points: [SplatScenePoint]) throws {
        print("adding points: \(points.count)")
        
        // Detect spherical harmonics capabilities
        updateSHCapabilities(for: points)
        
        do {
            try ensureAdditionalCapacity(points.count)
        } catch {
            Self.log.error("Failed to grow buffers: \(error)")
            return
        }

        splatBuffer.append(points.map { Splat($0) })
    }
    
    /// Update SH capabilities based on loaded point data
    private func updateSHCapabilities(for points: [SplatScenePoint]) {
        let hasSHData = points.contains { $0.hasSphericalHarmonics }
        let formatSupportsSH = points.contains { $0.formatSupportsSphericalHarmonics }
        
        // Update SH support status
        if hasSHData || formatSupportsSH {
            dataSupportsSphericalHarmonics = true
            
            // Log SH detection for debugging
            let shSplatsCount = points.filter { $0.hasSphericalHarmonics }.count
            let shPercentage = Float(shSplatsCount) / Float(points.count) * 100
            
            Self.log.info("SH capabilities detected: \(shSplatsCount)/\(points.count) splats (\(String(format: "%.1f", shPercentage))%) have SH data")
//            Self.log.info("Rendering mode: \(renderingMode), Will use SH shaders: \(useSHRendering)")
        } else {
            Self.log.info("No spherical harmonics data detected, using basic color rendering")
        }
    }

    public func add(_ point: SplatScenePoint) throws {
        try add([ point ])
    }

    private func switchToNextDynamicBuffer() {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxSimultaneousRenders
        uniformBufferOffset = UniformsArray.alignedSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents() + uniformBufferOffset).bindMemory(to: UniformsArray.self, capacity: 1)
    }

    private func updateUniforms(forViewports viewports: [ViewportDescriptor],
                                splatCount: UInt32,
                                indexedSplatCount: UInt32) {
        for (i, viewport) in viewports.enumerated() where i <= maxViewCount {
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }

        cameraWorldPosition = viewports.map { Self.cameraWorldPosition(forViewMatrix: $0.viewMatrix) }.mean ?? .zero
        cameraWorldForward = viewports.map { Self.cameraWorldForward(forViewMatrix: $0.viewMatrix) }.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        if !sorting {
            resort()
        }
    }

    private static func cameraWorldForward(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz
    }

    private static func cameraWorldPosition(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz
    }

    func renderEncoder(multiStage: Bool,
                       viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       for commandBuffer: MTLCommandBuffer) -> MTLRenderCommandEncoder {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = preserveExistingContent ? .load : .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        if let depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = preserveExistingContent ? .load : .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = preserveExistingContent ? 1.0 : 0.0
        }
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength

        renderPassDescriptor.tileWidth  = Constants.tileSize.width
        renderPassDescriptor.tileHeight = Constants.tileSize.height

        if multiStage {
            if let initializePipelineState {
                renderPassDescriptor.imageblockSampleLength = initializePipelineState.imageblockSampleLength
            } else {
                Self.log.error("initializePipeline == nil in renderEncoder()")
            }
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.setViewports(viewports.map(\.viewport))

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        return renderEncoder
    }
    

    public func render(viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        let splatCount = splatBuffer.count
        guard splatBuffer.count != 0 else { return }
        let indexedSplatCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + indexedSplatCount - 1) / indexedSplatCount

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(splatCount), indexedSplatCount: UInt32(indexedSplatCount))

        let multiStage = useMultiStagePipeline
        if multiStage {
            try buildMultiStagePipelineStatesIfNeeded()
        } else {
            try buildSingleStagePipelineStatesIfNeeded()
        }

        let renderEncoder = renderEncoder(multiStage: multiStage,
                                          viewports: viewports,
                                          colorTexture: colorTexture,
                                          colorStoreAction: colorStoreAction,
                                          depthTexture: depthTexture,
                                          rasterizationRateMap: rasterizationRateMap,
                                          renderTargetArrayLength: renderTargetArrayLength,
                                          for: commandBuffer)

        let indexCount = indexedSplatCount * 6
        if indexBuffer.count < indexCount {
            do {
                try indexBuffer.ensureCapacity(indexCount)
            } catch {
                return
            }
            indexBuffer.count = indexCount
            for i in 0..<indexedSplatCount {
                indexBuffer.values[i * 6 + 0] = UInt32(i * 4 + 0)
                indexBuffer.values[i * 6 + 1] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 2] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 3] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 4] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 5] = UInt32(i * 4 + 3)
            }
        }

        if multiStage {
            guard let initializePipelineState,
                  let drawSplatPipelineState
            else { return }

            renderEncoder.pushDebugGroup("Initialize")
            renderEncoder.setRenderPipelineState(initializePipelineState)
            renderEncoder.dispatchThreadsPerTile(Constants.tileSize)
            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(drawSplatPipelineState)
            renderEncoder.setDepthStencilState(drawSplatDepthState)
        } else {
            guard let singleStagePipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(singleStagePipelineState)
            renderEncoder.setDepthStencilState(singleStageDepthState)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)

        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                            indexCount: indexCount,
                                            indexType: .uint32,
                                            indexBuffer: indexBuffer.buffer,
                                            indexBufferOffset: 0,
                                            instanceCount: instanceCount)

        if multiStage {
            guard let postprocessPipelineState
            else { return }

            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Postprocess")
            renderEncoder.setRenderPipelineState(postprocessPipelineState)
            renderEncoder.setDepthStencilState(postprocessDepthState)
            renderEncoder.setCullMode(.none)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.popDebugGroup()
        } else {
            renderEncoder.popDebugGroup()
        }

        renderEncoder.endEncoding()
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort() {
        guard !sorting else { return }

        // Dispatch to appropriate sorting algorithm
        switch sortingAlgorithm {
        case .cpuStandard:
            cpuResort()
        case .gpuHybrid:
            if depthComputePipeline != nil && reorderPipeline != nil {
                gpuHybridResort()
            } else {
                cpuResort()
            }
        case .gpuBitonic:
            if depthComputePipeline != nil && bitonicSortPipeline != nil && reorderPipeline != nil {
                gpuBitonicResort()
            } else {
                cpuResort()
            }
        case .gpuRadix:
            // Radix sort is experimental, fall back to bitonic for now
            if depthComputePipeline != nil && bitonicSortPipeline != nil && reorderPipeline != nil {
                gpuBitonicResort()
            } else {
                cpuResort()
            }
        }
    }

    /// Benchmark all sorting algorithms and return results
    /// - Parameter iterations: Number of iterations per algorithm for averaging
    /// - Returns: Array of benchmark results for each algorithm
    public func benchmarkSortingAlgorithms(iterations: Int = 5) async -> [SortBenchmarkResult] {
        var results: [SortBenchmarkResult] = []
        let originalAlgorithm = sortingAlgorithm
        let originalBenchmarking = enableSortBenchmarking

        enableSortBenchmarking = true

        for algorithm in SortingAlgorithm.allCases {
            sortingAlgorithm = algorithm

            var totalTimes: [TimeInterval] = []

            for _ in 0..<iterations {
                let startTime = Date()

                // Force a resort
                sorting = false
                resort()

                // Wait for sort to complete
                while sorting {
                    try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
                }

                let elapsed = -startTime.timeIntervalSinceNow
                totalTimes.append(elapsed)
            }

            // Calculate average
            let avgTime = totalTimes.reduce(0, +) / Double(iterations)

            let result = SortBenchmarkResult(
                algorithm: algorithm,
                splatCount: splatBuffer.count,
                depthComputeTime: 0,  // Would need more detailed timing
                sortTime: avgTime,
                reorderTime: 0,
                totalTime: avgTime
            )
            results.append(result)

            print("Benchmark: \(result.description)")
        }

        // Restore original settings
        sortingAlgorithm = originalAlgorithm
        enableSortBenchmarking = originalBenchmarking

        return results
    }

    // ========================================================================
    // SORTING VALIDATION
    // ========================================================================

    /// Validates that the sort produced correct results (depths in descending order = far to near)
    /// Returns true if valid, false if there are sorting errors
    public func validateSortOrder() -> (isValid: Bool, errors: Int, firstErrorIndex: Int?) {
        let count = splatBuffer.count
        guard count > 1 else { return (true, 0, nil) }

        let cameraPos = cameraWorldPosition

        var errors = 0
        var firstErrorIndex: Int? = nil

        for i in 0..<(count - 1) {
            let pos1 = splatBuffer.values[i].position.simd
            let pos2 = splatBuffer.values[i + 1].position.simd

            let depth1 = (pos1 - cameraPos).lengthSquared
            let depth2 = (pos2 - cameraPos).lengthSquared

            // Should be far to near, so depth1 >= depth2
            if depth1 < depth2 {
                errors += 1
                if firstErrorIndex == nil {
                    firstErrorIndex = i
                }
            }
        }

        return (errors == 0, errors, firstErrorIndex)
    }

    /// Logs validation result
    private func logValidation(algorithm: String) {
        let (isValid, errors, firstError) = validateSortOrder()
        if isValid {
            print("✅ [\(algorithm)] Sort validation PASSED - all \(splatBuffer.count) splats correctly ordered")
        } else {
            print("❌ [\(algorithm)] Sort validation FAILED - \(errors) errors, first at index \(firstError ?? -1)")
        }
    }

    // GPU-accelerated sorting: GPU depth computation + CPU sort + GPU reorder
    // This hybrid approach is faster than pure CPU because:
    // - Depth computation: O(n) parallel on GPU vs O(n) serial on CPU
    // - Sort: O(n log n) on CPU but only sorting 8-byte structs (not 64-byte splats)
    // - Reorder: O(n) parallel on GPU vs O(n) serial memory copies on CPU
    private func gpuHybridResort() {
        guard !sorting else { return }
        sorting = true
        onSortStart?()
        let sortStartTime = Date()

        let splatCount = splatBuffer.count
        guard splatCount > 0,
              let commandQueue = commandQueue,
              let depthComputePipeline = depthComputePipeline,
              let reorderPipeline = reorderPipeline else {
            sorting = false
            return
        }

        // Ensure buffers are allocated
        ensureGPUSortBuffers(splatCount: splatCount)

        guard let depthBuffer = depthBuffer,
              let indexBufferForSort = indexBufferForSort,
              let sortedIndexBuffer = sortedIndexBuffer else {
            sorting = false
            cpuResort()  // Fallback to CPU
            return
        }

        // Capture camera state
        let cameraPos = cameraWorldPosition
        let cameraFwd = cameraWorldForward

        Task(priority: .high) {
            var depthComputeTime: TimeInterval = 0
            var sortTime: TimeInterval = 0
            var reorderTime: TimeInterval = 0

            defer {
                let totalTime = -sortStartTime.timeIntervalSinceNow
                sorting = false
                onSortComplete?(totalTime)

                // Log timing
                print("⏱️ [GPU Hybrid] Sorting \(splatCount) splats:")
                print("   Depth compute: \(String(format: "%.2f", depthComputeTime * 1000))ms")
                print("   CPU Sort:      \(String(format: "%.2f", sortTime * 1000))ms")
                print("   GPU Reorder:   \(String(format: "%.2f", reorderTime * 1000))ms")
                print("   TOTAL:         \(String(format: "%.2f", totalTime * 1000))ms")

                if enableSortBenchmarking {
                    logValidation(algorithm: "GPU Hybrid")
                }
            }

            // Step 1: Compute depths on GPU (parallel)
            let depthStart = Date()
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            commandBuffer.label = "GPU Depth Computation"

            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.label = "Depth Computation"
                computeEncoder.setComputePipelineState(depthComputePipeline)

                // Set buffers
                computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(depthBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(indexBufferForSort, offset: 0, index: 2)

                // Set uniforms
                var uniforms = DepthComputeUniforms(
                    cameraPosition: cameraPos,
                    cameraForward: cameraFwd,
                    splatCount: UInt32(splatCount),
                    sortByDistance: Constants.sortByDistance
                )
                computeEncoder.setBytes(&uniforms, length: MemoryLayout<DepthComputeUniforms>.stride, index: 3)

                // Dispatch
                let threadGroupSize = min(depthComputePipeline.maxTotalThreadsPerThreadgroup, 256)
                let threadGroups = (splatCount + threadGroupSize - 1) / threadGroupSize
                computeEncoder.dispatchThreadgroups(
                    MTLSize(width: threadGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
                )
                computeEncoder.endEncoding()
            }

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            depthComputeTime = -depthStart.timeIntervalSinceNow

            // Step 2: Sort on CPU using radix sort - O(n) instead of O(n log n)!
            let sortStart = Date()
            let depthPtr = depthBuffer.contents().bindMemory(to: Float.self, capacity: splatCount)
            let indexPtr = indexBufferForSort.contents().bindMemory(to: UInt32.self, capacity: splatCount)
            let sortedIndexPtr = sortedIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: splatCount)

            // Radix sort for floats: 4 passes (8 bits each) = O(4n) = O(n)
            // Convert floats to sortable integers using IEEE 754 trick
            radixSortFloatIndices(
                depths: depthPtr,
                indices: indexPtr,
                output: sortedIndexPtr,
                count: splatCount
            )
            sortTime = -sortStart.timeIntervalSinceNow

            // Step 3: Reorder splats on GPU (parallel memory copies)
            let reorderStart = Date()
            do {
                try splatBufferPrime.setCapacity(splatCount)
                splatBufferPrime.count = splatCount
            } catch {
                return
            }

            guard let reorderBuffer = commandQueue.makeCommandBuffer() else { return }
            reorderBuffer.label = "GPU Splat Reordering"

            if let computeEncoder = reorderBuffer.makeComputeCommandEncoder() {
                computeEncoder.label = "Splat Reordering"
                computeEncoder.setComputePipelineState(reorderPipeline)

                computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(splatBufferPrime.buffer, offset: 0, index: 1)
                computeEncoder.setBuffer(sortedIndexBuffer, offset: 0, index: 2)

                var count = UInt32(splatCount)
                computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)

                let threadGroupSize = min(reorderPipeline.maxTotalThreadsPerThreadgroup, 256)
                let threadGroups = (splatCount + threadGroupSize - 1) / threadGroupSize
                computeEncoder.dispatchThreadgroups(
                    MTLSize(width: threadGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
                )
                computeEncoder.endEncoding()
            }

            reorderBuffer.commit()
            reorderBuffer.waitUntilCompleted()
            reorderTime = -reorderStart.timeIntervalSinceNow

            // Swap buffers
            swap(&splatBuffer, &splatBufferPrime)
        }
    }

    // ========================================================================
    // GPU BITONIC SORT - Fully parallel sorting on GPU
    // ========================================================================
    // Bitonic sort has O(n * log²n) comparisons but all comparisons in each
    // pass are independent, making it highly parallel on GPU.
    //
    // For 100,000 splats: ~17 stages, ~153 passes total
    // Each pass processes all elements in parallel
    private func gpuBitonicResort() {
        guard !sorting else { return }
        sorting = true
        onSortStart?()
        let sortStartTime = Date()

        let splatCount = splatBuffer.count
        guard splatCount > 0,
              let commandQueue = commandQueue,
              let depthComputePipeline = depthComputePipeline,
              let bitonicSortPipeline = bitonicSortPipeline,
              let reorderPipeline = reorderPipeline else {
            sorting = false
            return
        }

        // Ensure buffers are allocated
        ensureGPUSortBuffers(splatCount: splatCount)

        guard let depthBuffer = depthBuffer,
              let indexBufferForSort = indexBufferForSort,
              let sortedIndexBuffer = sortedIndexBuffer else {
            sorting = false
            cpuResort()
            return
        }

        let cameraPos = cameraWorldPosition
        let cameraFwd = cameraWorldForward

        // Calculate padded size once for the whole operation
        let paddedCount = nextPowerOf2(splatCount)

        Task(priority: .high) {
            var depthComputeTime: TimeInterval = 0
            var bitonicSortTime: TimeInterval = 0
            var reorderTime: TimeInterval = 0
            var totalPasses = 0

            defer {
                let totalTime = -sortStartTime.timeIntervalSinceNow
                sorting = false
                onSortComplete?(totalTime)

                // Log timing
                let numStages = Int(log2(Double(paddedCount)))
                print("⏱️ [GPU Bitonic] Sorting \(splatCount) splats (padded to \(paddedCount), \(numStages) stages, \(totalPasses) passes):")
                print("   Depth compute:  \(String(format: "%.2f", depthComputeTime * 1000))ms")
                print("   Bitonic sort:   \(String(format: "%.2f", bitonicSortTime * 1000))ms")
                print("   GPU Reorder:    \(String(format: "%.2f", reorderTime * 1000))ms")
                print("   TOTAL:          \(String(format: "%.2f", totalTime * 1000))ms")

                // Always validate bitonic sort to ensure correctness
                logValidation(algorithm: "GPU Bitonic")
            }

            // Step 1: Compute depths on GPU
            let depthStart = Date()
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            commandBuffer.label = "GPU Bitonic Sort - Depth Computation"

            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.label = "Depth Computation"
                computeEncoder.setComputePipelineState(depthComputePipeline)

                computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(depthBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(indexBufferForSort, offset: 0, index: 2)

                var uniforms = DepthComputeUniforms(
                    cameraPosition: cameraPos,
                    cameraForward: cameraFwd,
                    splatCount: UInt32(splatCount),
                    sortByDistance: Constants.sortByDistance
                )
                computeEncoder.setBytes(&uniforms, length: MemoryLayout<DepthComputeUniforms>.stride, index: 3)

                let threadGroupSize = min(depthComputePipeline.maxTotalThreadsPerThreadgroup, 256)
                let threadGroups = (splatCount + threadGroupSize - 1) / threadGroupSize
                computeEncoder.dispatchThreadgroups(
                    MTLSize(width: threadGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
                )
                computeEncoder.endEncoding()
            }

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            depthComputeTime = -depthStart.timeIntervalSinceNow

            // Step 2: Bitonic Sort on GPU
            // Bitonic sort requires power-of-2 array size (paddedCount already calculated)
            let sortStart = Date()
            let numStages = Int(log2(Double(paddedCount)))

            // Initialize padding elements with +∞ depth (so they sort to the end)
            // OPTIMIZED: Use memset-style initialization instead of loop
            let depthPtr = depthBuffer.contents().bindMemory(to: Float.self, capacity: paddedCount)
            let indexPtr = indexBufferForSort.contents().bindMemory(to: UInt32.self, capacity: paddedCount)
            let paddingCount = paddedCount - splatCount
            if paddingCount > 0 {
                // Fill padding depths with infinity using fast memory operations
                let paddingDepthPtr = depthPtr.advanced(by: splatCount)
                let paddingIndexPtr = indexPtr.advanced(by: splatCount)

                // Use vDSP for fast fill if available, otherwise stride-based fill
                let infValue = Float.infinity
                vDSP_vfill([infValue], paddingDepthPtr, 1, vDSP_Length(paddingCount))

                // Fill indices with sequential values starting from splatCount
                for i in 0..<paddingCount {
                    paddingIndexPtr[i] = UInt32(splatCount + i)
                }
            }

            // Run bitonic sort on the full padded array
            // Optimization: batch multiple passes per command buffer to reduce overhead
            // Each pass needs a memory barrier, but we can batch passes that don't conflict
            let passesPerBatch = 10  // Commit every 10 passes to balance overhead vs latency
            var passesInCurrentBatch = 0
            var currentCommandBuffer: MTLCommandBuffer?
            let threadGroupSize = min(bitonicSortPipeline.maxTotalThreadsPerThreadgroup, 256)
            let threadGroups = (paddedCount + threadGroupSize - 1) / threadGroupSize

            for stage in 1...numStages {
                let stageDistance = UInt32(1 << stage)

                // Each stage has 'stage' number of passes
                for pass in stride(from: stage, through: 1, by: -1) {
                    let passDistance = UInt32(1 << (pass - 1))
                    totalPasses += 1

                    // Start a new command buffer if needed
                    if currentCommandBuffer == nil {
                        currentCommandBuffer = commandQueue.makeCommandBuffer()
                        currentCommandBuffer?.label = "Bitonic Sort Batch"
                        passesInCurrentBatch = 0
                    }

                    guard let sortBuffer = currentCommandBuffer else { return }

                    // Each pass needs its own compute encoder for proper synchronization
                    if let computeEncoder = sortBuffer.makeComputeCommandEncoder() {
                        computeEncoder.setComputePipelineState(bitonicSortPipeline)

                        computeEncoder.setBuffer(depthBuffer, offset: 0, index: 0)
                        computeEncoder.setBuffer(indexBufferForSort, offset: 0, index: 1)

                        // Use paddedCount for sorting, so all elements participate
                        var sortUniforms = SortUniforms(
                            count: UInt32(paddedCount),  // Use padded count!
                            stageDistance: stageDistance,
                            passDistance: passDistance,
                            ascending: 1  // Ascending sort (depths are negated for far-to-near)
                        )
                        computeEncoder.setBytes(&sortUniforms, length: MemoryLayout<SortUniforms>.stride, index: 2)

                        computeEncoder.dispatchThreadgroups(
                            MTLSize(width: threadGroups, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
                        )
                        computeEncoder.endEncoding()
                    }

                    passesInCurrentBatch += 1

                    // Commit batch when full or at end of all passes
                    let isLastPass = (stage == numStages && pass == 1)
                    if passesInCurrentBatch >= passesPerBatch || isLastPass {
                        sortBuffer.commit()
                        sortBuffer.waitUntilCompleted()
                        currentCommandBuffer = nil
                    }
                }
            }
            bitonicSortTime = -sortStart.timeIntervalSinceNow

            // Copy sorted indices to output buffer
            // OPTIMIZED: Since padding depths are +∞, they sort to the END
            // So the first splatCount indices are guaranteed to be valid - use direct memcpy!
            let sortedIndexPtr = indexBufferForSort.contents()
            let dstPtr = sortedIndexBuffer.contents()

            // Direct memory copy - much faster than element-by-element loop
            memcpy(dstPtr, sortedIndexPtr, splatCount * MemoryLayout<UInt32>.stride)

            // Step 3: Reorder splats based on sorted indices
            let reorderStart = Date()
            do {
                try splatBufferPrime.setCapacity(splatCount)
                splatBufferPrime.count = splatCount
            } catch {
                return
            }

            guard let reorderBuffer = commandQueue.makeCommandBuffer() else { return }
            reorderBuffer.label = "GPU Splat Reordering"

            if let computeEncoder = reorderBuffer.makeComputeCommandEncoder() {
                computeEncoder.label = "Splat Reordering"
                computeEncoder.setComputePipelineState(reorderPipeline)

                computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(splatBufferPrime.buffer, offset: 0, index: 1)
                computeEncoder.setBuffer(sortedIndexBuffer, offset: 0, index: 2)

                var count = UInt32(splatCount)
                computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)

                let threadGroupSize = min(reorderPipeline.maxTotalThreadsPerThreadgroup, 256)
                let threadGroups = (splatCount + threadGroupSize - 1) / threadGroupSize
                computeEncoder.dispatchThreadgroups(
                    MTLSize(width: threadGroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
                )
                computeEncoder.endEncoding()
            }

            reorderBuffer.commit()
            reorderBuffer.waitUntilCompleted()
            reorderTime = -reorderStart.timeIntervalSinceNow

            // Swap buffers
            swap(&splatBuffer, &splatBufferPrime)
        }
    }

    // Original CPU-based sorting (fallback)
    private func cpuResort() {
        guard !sorting else { return }
        sorting = true
        onSortStart?()
        let sortStartTime = Date()

        let splatCount = splatBuffer.count

        let cameraWorldForward = cameraWorldForward
        let cameraWorldPosition = cameraWorldPosition

        Task(priority: .high) {
            var depthComputeTime: TimeInterval = 0
            var sortTime: TimeInterval = 0
            var reorderTime: TimeInterval = 0

            defer {
                let totalTime = -sortStartTime.timeIntervalSinceNow
                sorting = false
                onSortComplete?(totalTime)

                // Log timing
                print("⏱️ [CPU Standard] Sorting \(splatCount) splats:")
                print("   Depth compute: \(String(format: "%.2f", depthComputeTime * 1000))ms")
                print("   CPU Sort:      \(String(format: "%.2f", sortTime * 1000))ms")
                print("   CPU Reorder:   \(String(format: "%.2f", reorderTime * 1000))ms")
                print("   TOTAL:         \(String(format: "%.2f", totalTime * 1000))ms")

                if enableSortBenchmarking {
                    logValidation(algorithm: "CPU Standard")
                }
            }

            // Step 1: Compute depths (serial on CPU)
            let depthStart = Date()
            if orderAndDepthTempSort.count != splatCount {
                orderAndDepthTempSort = Array(repeating: SplatIndexAndDepth(index: .max, depth: 0), count: splatCount)
            }

            if Constants.sortByDistance {
                for i in 0..<splatCount {
                    orderAndDepthTempSort[i].index = UInt32(i)
                    let splatPosition = splatBuffer.values[i].position.simd
                    orderAndDepthTempSort[i].depth = (splatPosition - cameraWorldPosition).lengthSquared
                }
            } else {
                for i in 0..<splatCount {
                    orderAndDepthTempSort[i].index = UInt32(i)
                    let splatPosition = splatBuffer.values[i].position.simd
                    orderAndDepthTempSort[i].depth = dot(splatPosition, cameraWorldForward)
                }
            }
            depthComputeTime = -depthStart.timeIntervalSinceNow

            // Step 2: Sort (CPU)
            let sortStart = Date()
            orderAndDepthTempSort.sort { $0.depth > $1.depth }
            sortTime = -sortStart.timeIntervalSinceNow

            // Step 3: Reorder (serial CPU memory copies)
            let reorderStart = Date()
            do {
                try splatBufferPrime.setCapacity(splatCount)
                splatBufferPrime.count = 0
                for newIndex in 0..<orderAndDepthTempSort.count {
                    let oldIndex = Int(orderAndDepthTempSort[newIndex].index)
                    splatBufferPrime.append(splatBuffer, fromIndex: oldIndex)
                }

                swap(&splatBuffer, &splatBufferPrime)
            } catch {
                print("❌ [CPU Standard] Reorder failed: \(error)")
            }
            reorderTime = -reorderStart.timeIntervalSinceNow
        }
    }
    
    // SPZ-specific color correction function
    private static func applySPZColorCorrection(_ color: SIMD3<Float>) -> SIMD3<Float> {
        let brightness = SPZColorSettings.brightness
        let gamma = SPZColorSettings.gamma
        return SIMD3<Float>(
            min(1.0, color.x),
            min(1.0, color.y),
            min(1.0, color.z)
        )
//        return SIMD3<Float>(
//            min(1.0, pow(color.x, gamma) * brightness),
//            min(1.0, pow(color.y, gamma) * brightness),
//            min(1.0, pow(color.z, gamma) * brightness)
//        )
    }
}

extension SplatRenderer.Splat {
    init(_ splat: SplatScenePoint) {
        // Handle color processing based on source format capabilities
        var colorRGB: SIMD3<Float>
        
        // Use appropriate color conversion based on format
        switch splat.sourceCapabilities.formatName {
        case let name where name.contains(".spz"):
            // SPZ files: Use configured color correction
            colorRGB = SplatRenderer.applySPZColorCorrection(splat.color.asLinearFloat)
        case let name where name.contains(".splat"):
            // .splat files: Apply sRGB to linear conversion
            colorRGB = splat.color.asLinearFloat.sRGBToLinear
        default:
            // PLY files and others: Use linear color directly
            colorRGB = splat.color.asLinearFloat
        }
        
        self.init(position: splat.position,
                  color: .init(colorRGB, splat.opacity.asLinearFloat),
                  scale: splat.scale.asLinearFloat,
                  rotation: splat.rotation.normalized,
                  isSpz: splat.isSpz)
    }

    init(position: SIMD3<Float>,
         color: SIMD4<Float>,
         scale: SIMD3<Float>,
         rotation: simd_quatf,
         isSpz: Bool) {
        let transform = simd_float3x3(rotation) * simd_float3x3(diagonal: scale)
        var cov3D = transform * transform.transpose
        
        if isSpz {
            let scaleFactor: Float = 0.01
            cov3D = cov3D * scaleFactor
        }
        
        self.init(position: MTLPackedFloat3Make(position.x, position.y, position.z),
                  color: SplatRenderer.PackedRGBHalf4(r: Float16(color.x), g: Float16(color.y), b: Float16(color.z), a: Float16(color.w)),
                  covA: SplatRenderer.PackedHalf3(x: Float16(cov3D[0, 0]), y: Float16(cov3D[0, 1]), z: Float16(cov3D[0, 2])),
                  covB: SplatRenderer.PackedHalf3(x: Float16(cov3D[1, 1]), y: Float16(cov3D[1, 2]), z: Float16(cov3D[2, 2])))
    }
}

protocol MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { get }
}

extension UInt32: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint32 }
}
extension UInt16: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint16 }
}

extension Array where Element == SIMD3<Float> {
    var mean: SIMD3<Float>? {
        guard !isEmpty else { return nil }
        return reduce(.zero, +) / Float(count)
    }
}

private extension MTLPackedFloat3 {
    var simd: SIMD3<Float> {
        SIMD3(x: x, y: y, z: z)
    }
}

private extension SIMD3 where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    var normalized: SIMD3<Scalar> {
        self / Scalar(sqrt(lengthSquared))
    }

    var lengthSquared: Scalar {
        x*x + y*y + z*z
    }

    func vector4(w: Scalar) -> SIMD4<Scalar> {
        SIMD4<Scalar>(x: x, y: y, z: z, w: w)
    }

    static func random(in range: Range<Scalar>) -> SIMD3<Scalar> {
        Self(x: Scalar.random(in: range), y: .random(in: range), z: .random(in: range))
    }
}

private extension SIMD3<Float> {
    var sRGBToLinear: SIMD3<Float> {
        SIMD3(x: pow(x, 2.2), y: pow(y, 2.2), z: pow(z, 2.2))
    }
}

private extension SIMD4 where Scalar: BinaryFloatingPoint {
    var xyz: SIMD3<Scalar> {
        .init(x: x, y: y, z: z)
    }
}

private extension MTLLibrary {
    func makeRequiredFunction(name: String) -> MTLFunction {
        guard let result = makeFunction(name: name) else {
            fatalError("Unable to load required shader function: \"\(name)\"")
        }
        return result
    }
}
