#if os(iOS)

import Foundation
import Metal
import MetalKit
import ARKit
import MetalSplatter
import SampleBoxRenderer
import SplatIO
import simd
import os

public class ARSplatRenderer: NSObject {
    private static let log = Logger(
        subsystem: Bundle.module.bundleIdentifier!,
        category: "ARSplatRenderer"
    )
    
    public let device: MTLDevice
    public let colorFormat: MTLPixelFormat
    public let depthFormat: MTLPixelFormat
    public let sampleCount: Int
    public let maxSimultaneousRenders: Int
    
    private let coreSplatRenderer: SplatRenderer
    private let arCameraRenderer: ARCameraRenderer
    private let inFlightSemaphore: DispatchSemaphore
    
    public let arSession: ARSession
    
    // AR-specific configuration
    public var isAREnabled: Bool = false {
        didSet {
            updateARSession()
        }
    }
    
    // Splat positioning in AR space
    public var splatScale: Float = 1.0
    public var splatPosition: SIMD3<Float> = SIMD3(0, 0, -1)
    public var splatRotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    
    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
        self.device = device
        self.colorFormat = colorFormat
        self.depthFormat = depthFormat
        self.sampleCount = sampleCount
        self.maxSimultaneousRenders = maxSimultaneousRenders
        
        self.coreSplatRenderer = try SplatRenderer(
            device: device,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: sampleCount,
            maxViewCount: maxViewCount,
            maxSimultaneousRenders: maxSimultaneousRenders
        )
        
        guard let arCameraRenderer = ARCameraRenderer(device: device) else {
            throw NSError(domain: "ARSplatRenderer", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create AR camera renderer"])
        }
        self.arCameraRenderer = arCameraRenderer
        
        self.arSession = ARSession()
        self.inFlightSemaphore = DispatchSemaphore(value: maxSimultaneousRenders)
        
        super.init()
        
        arSession.delegate = self
    }
    
    private func updateARSession() {
        if isAREnabled {
            let configuration = ARWorldTrackingConfiguration()
            configuration.planeDetection = [.horizontal, .vertical]
            configuration.environmentTexturing = .automatic
            
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                configuration.sceneReconstruction = .mesh
            }
            
            arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            Self.log.info("AR Session started with world tracking")
        } else {
            arSession.pause()
            Self.log.info("AR Session paused")
        }
    }
    
    // MARK: - Core SplatRenderer Interface
    
    public func reset() {
        coreSplatRenderer.reset()
    }
    
    public func read(from url: URL) async throws {
        try await coreSplatRenderer.read(from: url)
    }
    
    public func add(_ points: [SplatScenePoint]) throws {
        try coreSplatRenderer.add(points)
    }
    
    public func add(_ point: SplatScenePoint) throws {
        try coreSplatRenderer.add(point)
    }
    
    public var splatCount: Int {
        coreSplatRenderer.splatCount
    }
    
    public var clearColor: MTLClearColor {
        get { coreSplatRenderer.clearColor }
        set { coreSplatRenderer.clearColor = newValue }
    }
    
    // MARK: - AR-Enhanced Rendering
    
    public func render(viewports: [SplatRenderer.ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { _ in
            semaphore.signal()
        }
        
        if isAREnabled, let currentFrame = arSession.currentFrame {
            try renderARComposition(
                frame: currentFrame,
                colorTexture: colorTexture,
                colorStoreAction: colorStoreAction,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                renderTargetArrayLength: renderTargetArrayLength,
                to: commandBuffer
            )
        } else {
            // Fallback to regular splat rendering
            try coreSplatRenderer.render(
                viewports: viewports,
                colorTexture: colorTexture,
                colorStoreAction: colorStoreAction,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                renderTargetArrayLength: renderTargetArrayLength,
                to: commandBuffer
            )
        }
    }
    
    private func renderARComposition(
        frame: ARFrame,
        colorTexture: MTLTexture,
        colorStoreAction: MTLStoreAction,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        renderTargetArrayLength: Int,
        to commandBuffer: MTLCommandBuffer
    ) throws {
        
        // Create render pass descriptor for composition
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        
        if let depthTexture = depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }
        
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw NSError(domain: "ARSplatRenderer", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create render command encoder"])
        }
        
        renderEncoder.label = "AR Splat Composition"
        
        // Phase 1: Render AR camera background
        arCameraRenderer.render(frame: frame, to: renderEncoder)
        
        renderEncoder.endEncoding()
        
        // Phase 2: Render splats with AR camera matrices
        let arViewport = createARViewport(from: frame, colorTexture: colorTexture)
        
        try coreSplatRenderer.render(
            viewports: [arViewport],
            colorTexture: colorTexture,
            colorStoreAction: colorStoreAction,
            depthTexture: depthTexture,
            rasterizationRateMap: rasterizationRateMap,
            renderTargetArrayLength: renderTargetArrayLength,
            to: commandBuffer
        )
    }
    
    private func createARViewport(from frame: ARFrame, colorTexture: MTLTexture) -> SplatRenderer.ViewportDescriptor {
        let camera = frame.camera
        let textureSize = CGSize(width: colorTexture.width, height: colorTexture.height)
        
        let viewport = MTLViewport(
            originX: 0, originY: 0,
            width: Double(textureSize.width), height: Double(textureSize.height),
            znear: 0, zfar: 1
        )
        
        // Use ARKit's projection matrix
        let projectionMatrix = camera.projectionMatrix(
            for: .landscapeRight,
            viewportSize: textureSize,
            zNear: 0.01, zFar: 100.0
        )
        
        // Create model transform for splat positioning
        let rotationTransform = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))
        let scaleMatrix = matrix4x4_scale(splatScale, splatScale, splatScale)
        let translationMatrix = matrix4x4_translation(splatPosition.x, splatPosition.y, splatPosition.z)
        let rotationMatrix = simd_float4x4(splatRotation)
        
        let modelMatrix = translationMatrix * rotationMatrix * scaleMatrix * rotationTransform
        let viewMatrix = camera.viewMatrix(for: .landscapeRight) * modelMatrix
        
        return SplatRenderer.ViewportDescriptor(
            viewport: viewport,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            screenSize: SIMD2(x: Int(textureSize.width), y: Int(textureSize.height))
        )
    }
    
    public func resetSplatTransform() {
        splatScale = 1.0
        splatPosition = SIMD3(0, 0, -1)
        splatRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }
}

// MARK: - ARSessionDelegate

extension ARSplatRenderer: ARSessionDelegate {
    public func session(_ session: ARSession, didFailWithError error: Error) {
        Self.log.error("AR Session failed: \(error.localizedDescription)")
        isAREnabled = false
    }
    
    public func sessionWasInterrupted(_ session: ARSession) {
        Self.log.info("AR Session was interrupted")
    }
    
    public func sessionInterruptionEnded(_ session: ARSession) {
        Self.log.info("AR Session interruption ended")
    }
    
    public func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        switch camera.trackingState {
        case .normal:
            Self.log.info("AR Camera tracking: Normal")
        case .notAvailable:
            Self.log.warning("AR Camera tracking: Not Available")
        case .limited(let reason):
            Self.log.warning("AR Camera tracking limited: \(String(describing: reason))")
        }
    }
}

// MARK: - Matrix Utility Functions

fileprivate func matrix4x4_scale(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    return simd_float4x4(
        SIMD4<Float>(x, 0, 0, 0),
        SIMD4<Float>(0, y, 0, 0),
        SIMD4<Float>(0, 0, z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    )
}

fileprivate func matrix4x4_translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    return simd_float4x4(
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(x, y, z, 1)
    )
}

fileprivate func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> simd_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    
    return simd_float4x4(
        SIMD4<Float>(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
        SIMD4<Float>(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
        SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
        SIMD4<Float>(                  0,                   0,                   0, 1)
    )
}

#endif
