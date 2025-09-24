#if os(iOS)

import Foundation
import Metal
import MetalKit
import ARKit
import AVFoundation
import UIKit
import os

public class ARCameraRenderer {
    private static let log = Logger(
        subsystem: Bundle.module.bundleIdentifier!,
        category: "ARCameraRenderer"
    )
    
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelineState: MTLRenderPipelineState?
    private var vertexBuffer: MTLBuffer?
    private var textureCache: CVMetalTextureCache!
    private var depthStencilState: MTLDepthStencilState?
    private var transformBuffer: MTLBuffer?
    
    private struct Vertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
    }
    
    private struct CameraTransform {
        let displayTransform: simd_float3x3
    }
    
    // Base texture coordinates - will be transformed based on device orientation
    private let quadVertices: [Vertex] = [
        Vertex(position: SIMD2(-1, -1), texCoord: SIMD2(1, 1)),  // Bottom-left → Top-left of texture (Y-flipped)
        Vertex(position: SIMD2( 1, -1), texCoord: SIMD2(1, 0)),  // Bottom-right → Top-right of texture (Y-flipped)
        Vertex(position: SIMD2(-1,  1), texCoord: SIMD2(0, 1)),  // Top-left → Bottom-left of texture (Y-flipped)
        Vertex(position: SIMD2( 1,  1), texCoord: SIMD2(0, 0))   // Top-right → Bottom-right of texture (Y-flipped)
    ]
    
    public init?(device: MTLDevice) {
        self.device = device
        
        Self.log.info("Initializing ARCameraRenderer with device: \(device.name)")
        
        do {
            self.library = try device.makeDefaultLibrary(bundle: Bundle.module)
            Self.log.info("Metal library loaded successfully")
        } catch {
            Self.log.error("Failed to create MetalSplatter library: \(error)")
            print("ARCameraRenderer init failed: library error - \(error)")
            return nil
        }
        
        let textureCacheResult = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        guard textureCacheResult == kCVReturnSuccess else {
            Self.log.error("Failed to create CVMetalTextureCache with result: \(textureCacheResult)")
            print("ARCameraRenderer init failed: texture cache error - \(textureCacheResult)")
            return nil
        }
        Self.log.info("CVMetalTextureCache created successfully")
        
        setupPipelineState()
        setupVertexBuffer()
        setupDepthState()
        setupTransformBuffer()
        
        if pipelineState == nil {
            Self.log.error("Pipeline state is nil after setup")
            print("ARCameraRenderer init failed: pipeline state is nil")
            return nil
        }
        
        if vertexBuffer == nil {
            Self.log.error("Vertex buffer is nil after setup")
            print("ARCameraRenderer init failed: vertex buffer is nil")
            return nil
        }
        
        Self.log.info("ARCameraRenderer initialized successfully")
    }
    
    private func setupPipelineState() {
        Self.log.info("Setting up AR camera pipeline state")
        
        guard let vertexFunction = library.makeFunction(name: "arCameraVertexShader") else {
            Self.log.error("Failed to load arCameraVertexShader")
            return
        }
        
        guard let fragmentFunction = library.makeFunction(name: "arCameraFragmentShader") else {
            Self.log.error("Failed to load arCameraFragmentShader")
            return
        }
        
        Self.log.info("Loaded shader functions successfully")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false
        
        // CRITICAL FIX: Set the depth format to match the framebuffer
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        print("Pipeline descriptor configured:")
        print("   Color format: \(pipelineDescriptor.colorAttachments[0].pixelFormat)")
        print("   Depth format: \(pipelineDescriptor.depthAttachmentPixelFormat)")
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            Self.log.info("AR camera pipeline state created successfully")
        } catch {
            Self.log.error("Failed to create AR camera pipeline state: \(error)")
        }
    }
    
    private func setupVertexBuffer() {
        vertexBuffer = device.makeBuffer(
            bytes: quadVertices,
            length: quadVertices.count * MemoryLayout<Vertex>.stride,
            options: .storageModeShared
        )
        vertexBuffer?.label = "AR Camera Quad Vertices"
    }
    
    private func setupDepthState() {
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .equal  // Only render where depth == 0.0 (no splats)
        depthDescriptor.isDepthWriteEnabled = false    // Don't write depth, preserve splat depths
        
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
        Self.log.info("AR camera depth stencil state created - renders only where depth == 0.0")
    }
    
    private func setupTransformBuffer() {
        transformBuffer = device.makeBuffer(
            length: MemoryLayout<CameraTransform>.stride,
            options: .storageModeShared
        )
        transformBuffer?.label = "AR Camera Transform Buffer"
    }
    
    public func render(
        frame: ARFrame,
        to renderEncoder: MTLRenderCommandEncoder
    ) {
        render(frame: frame, viewportSize: CGSize(width: 1, height: 1), interfaceOrientation: .portrait, to: renderEncoder)
    }
    
    public func render(
        frame: ARFrame,
        viewportSize: CGSize,
        interfaceOrientation: UIInterfaceOrientation,
        to renderEncoder: MTLRenderCommandEncoder
    ) {
        Self.log.info("AR camera render called")
        
        guard let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer,
              let transformBuffer = transformBuffer else {
            Self.log.error("AR camera renderer not properly initialized - pipelineState: \(self.pipelineState != nil), vertexBuffer: \(self.vertexBuffer != nil), transformBuffer: \(self.transformBuffer != nil)")
            return
        }
        
        // Calculate display transform based on device orientation and viewport
        let cgTransform = frame.displayTransform(for: interfaceOrientation, viewportSize: viewportSize)
        
        // Debug the transform values
        print("🔧 Display transform: a=\(cgTransform.a), b=\(cgTransform.b), c=\(cgTransform.c), d=\(cgTransform.d), tx=\(cgTransform.tx), ty=\(cgTransform.ty)")
        print("📐 Viewport: \(viewportSize), Orientation: \(interfaceOrientation)")
        
        // TEMPORARY: Use identity transform to test
        let identityTransform = simd_float3x3(
            simd_float3(1, 0, 0),  // Column 1
            simd_float3(0, 1, 0),  // Column 2
            simd_float3(0, 0, 1)   // Column 3
        )
        
        // Convert CGAffineTransform to simd_float3x3 (column-major)
        let displayTransform = simd_float3x3(
            simd_float3(Float(cgTransform.a), Float(cgTransform.b), 0),  // Column 1
            simd_float3(Float(cgTransform.c), Float(cgTransform.d), 0),  // Column 2
            simd_float3(Float(cgTransform.tx), Float(cgTransform.ty), 1) // Column 3
        )
        
        // Use identity transform for now to test
        let cameraTransform = CameraTransform(displayTransform: identityTransform)
        
        // Update transform buffer
        let transformPointer = transformBuffer.contents().bindMemory(to: CameraTransform.self, capacity: 1)
        transformPointer.pointee = cameraTransform
        
        renderEncoder.pushDebugGroup("AR Camera Background")
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(transformBuffer, offset: 0, index: 1)
        
        // Set depth stencil state for proper depth writing
        if let depthStencilState = depthStencilState {
            renderEncoder.setDepthStencilState(depthStencilState)
            print("🎯 AR camera depth stencil state set")
        }
        
        // Log camera texture dimensions for debugging aspect ratio issues
        let capturedImage = frame.capturedImage
        let cameraWidth = CVPixelBufferGetWidth(capturedImage)
        let cameraHeight = CVPixelBufferGetHeight(capturedImage)
        print("📷 Camera texture dimensions: \(cameraWidth) x \(cameraHeight)")
        print("📷 Camera aspect ratio: \(Float(cameraWidth) / Float(cameraHeight))")
        
        let pixelFormat = CVPixelBufferGetPixelFormatType(capturedImage)
        Self.log.info("Captured image pixel format: \(pixelFormat)")
        
        if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
           pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            Self.log.info("Rendering YUV frame")
            renderYUVFrame(capturedImage, renderEncoder: renderEncoder)
        } else {
            Self.log.info("Rendering RGB frame")
            renderRGBFrame(capturedImage, renderEncoder: renderEncoder)
        }
        
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.popDebugGroup()
        Self.log.info("AR camera render completed")
    }
    
    private func renderYUVFrame(_ pixelBuffer: CVPixelBuffer, renderEncoder: MTLRenderCommandEncoder) {
        var yTexture: CVMetalTexture?
        var uvTexture: CVMetalTexture?
        
        let yWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let uvWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uvHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        
        print("Creating Y texture: \(yWidth)x\(yHeight)")
        let yResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .r8Unorm, yWidth, yHeight, 0, &yTexture
        )
        print("Y texture creation result: \(yResult)")
        
        print("Creating UV texture: \(uvWidth)x\(uvHeight)")
        let uvResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .rg8Unorm, uvWidth, uvHeight, 1, &uvTexture
        )
        print("UV texture creation result: \(uvResult)")
        
        if let yTexture = yTexture, let uvTexture = uvTexture,
           let yMetalTexture = CVMetalTextureGetTexture(yTexture),
           let uvMetalTexture = CVMetalTextureGetTexture(uvTexture) {
            print("Setting Y texture: \(yMetalTexture.width)x\(yMetalTexture.height)")
            print("Setting UV texture: \(uvMetalTexture.width)x\(uvMetalTexture.height)")
            renderEncoder.setFragmentTexture(yMetalTexture, index: 0)
            renderEncoder.setFragmentTexture(uvMetalTexture, index: 1)
        } else {
            print("FAILED to create Metal textures from camera frame!")
            print("yTexture: \(yTexture != nil), uvTexture: \(uvTexture != nil)")
        }
    }
    
    private func renderRGBFrame(_ pixelBuffer: CVPixelBuffer, renderEncoder: MTLRenderCommandEncoder) {
        var texture: CVMetalTexture?
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .bgra8Unorm, width, height, 0, &texture
        )
        
        if let texture = texture,
           let metalTexture = CVMetalTextureGetTexture(texture) {
            renderEncoder.setFragmentTexture(metalTexture, index: 0)
        }
    }
}

#endif
