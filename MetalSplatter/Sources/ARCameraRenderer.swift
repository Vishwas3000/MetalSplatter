#if os(iOS)

import Foundation
import Metal
import MetalKit
import ARKit
import AVFoundation
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
    
    private struct Vertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
    }
    
    // Portrait orientation: rotate 90° clockwise for correct camera orientation
    private let quadVertices: [Vertex] = [
        Vertex(position: SIMD2(-1, -1), texCoord: SIMD2(0, 0)),  // Bottom-left
        Vertex(position: SIMD2( 1, -1), texCoord: SIMD2(1, 0)),  // Bottom-right
        Vertex(position: SIMD2(-1,  1), texCoord: SIMD2(0, 1)),  // Top-left
        Vertex(position: SIMD2( 1,  1), texCoord: SIMD2(1, 1))   // Top-right
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
    
    public func render(
        frame: ARFrame,
        to renderEncoder: MTLRenderCommandEncoder
    ) {
        Self.log.info("AR camera render called")
        
        guard let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer else {
            Self.log.error("AR camera renderer not properly initialized - pipelineState: \(self.pipelineState != nil), vertexBuffer: \(self.vertexBuffer != nil)")
            return
        }
        
        renderEncoder.pushDebugGroup("AR Camera Background")
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        // Set depth stencil state for proper depth writing
        if let depthStencilState = depthStencilState {
            renderEncoder.setDepthStencilState(depthStencilState)
            print("🎯 AR camera depth stencil state set")
        }
        
        let capturedImage = frame.capturedImage
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
