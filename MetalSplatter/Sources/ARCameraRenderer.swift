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
        let cropScale: simd_float2
        let cropOffset: simd_float2
    }
    
    // Standard full-screen quad with normalized texture coordinates
    private let quadVertices: [Vertex] = [
        Vertex(position: SIMD2(-1, -1), texCoord: SIMD2(1, 0)),  // Bottom-left
        Vertex(position: SIMD2( 1, -1), texCoord: SIMD2(0, 0)),  // Bottom-right
        Vertex(position: SIMD2(-1,  1), texCoord: SIMD2(1, 1)),  // Top-left
        Vertex(position: SIMD2( 1,  1), texCoord: SIMD2(0, 1))   // Top-right
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
        
        // Set depth format to match framebuffer for direct rendering
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
        depthDescriptor.depthCompareFunction = .always  // Always render camera background
        depthDescriptor.isDepthWriteEnabled = true      // Write max depth so splats render in front
        
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
        Self.log.info("AR camera depth stencil state created - writes max depth for background")
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
        // Use camera image size as fallback viewport
        let capturedImage = frame.capturedImage
        let cameraWidth = CVPixelBufferGetWidth(capturedImage)
        let cameraHeight = CVPixelBufferGetHeight(capturedImage)
        let fallbackViewportSize = CGSize(width: cameraWidth, height: cameraHeight)
        
        render(frame: frame, viewportSize: fallbackViewportSize, interfaceOrientation: .portrait, to: renderEncoder)
    }
    
    /// Calculates centered texture offset for a given crop scale
    /// - Parameter cropScale: The scale factor for texture coordinates (e.g., 1.5 for 150% zoom)
    /// - Returns: Offset to center the cropped texture area
    private func calculateCenterOffset(for cropScale: simd_float2) -> simd_float2 {
        // Formula: offset = (1.0 - scale) * 0.5
        // This centers the scaled texture coordinates within [0,1] bounds
        return (simd_float2(1.0, 1.0) - cropScale) * 0.5
    }
    
    /// Calculates aspect-fill crop scale to eliminate stretching
    /// - Parameters:
    ///   - cameraAspectRatio: Aspect ratio of the camera texture (width/height)
    ///   - viewportAspectRatio: Aspect ratio of the viewport (width/height)
    /// - Returns: Crop scale factors for X and Y to achieve aspect-fill behavior
    private func calculateAspectFillCropScale(cameraAspectRatio: Float, viewportAspectRatio: Float) -> Float {
        var cropScale: Float = 1.0
        
        if cameraAspectRatio > viewportAspectRatio {
            // Camera is wider than viewport - crop horizontally (sides)
            // Scale down X to fit viewport aspect ratio
            cropScale = viewportAspectRatio / cameraAspectRatio
        } else {
            // Camera is taller than viewport - crop vertically (top/bottom)
            // Scale down Y to fit viewport aspect ratio  
            cropScale = cameraAspectRatio / viewportAspectRatio
        }
        
        return cropScale
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
        
        // Validate viewport size
        guard viewportSize.width > 0.0 && viewportSize.height > 0.0 else {
//            Self.log.error("Invalid viewport size: \(viewportSize.width)x\(viewportSize.height)")
            return
        }
        
        // Get camera image dimensions for aspect ratio calculation
        let capturedImage = frame.capturedImage
        let cameraWidth = CVPixelBufferGetWidth(capturedImage)
        let cameraHeight = CVPixelBufferGetHeight(capturedImage)
        let cameraAspectRatio = Float(cameraWidth) / Float(cameraHeight)
        let viewportAspectRatio = Float(viewportSize.width) / Float(viewportSize.height)
        
        Self.log.info("Camera: \(cameraWidth)x\(cameraHeight) (aspect: \(cameraAspectRatio)), Viewport: \(viewportSize.width)x\(viewportSize.height) (aspect: \(viewportAspectRatio))")
        
        print("🎯 REVERTING TO SIMPLE ARKIT DISPLAYTRANSFORM - FIXING BLEEDING LINES")
        
        // REVERT: Use ARKit's full displayTransform to fix bleeding lines issue
        // The bleeding was caused by our manual cropping going outside texture bounds
        let cgDisplayTransform = frame.displayTransform(for: interfaceOrientation, viewportSize: viewportSize)
        
        // Convert full CGAffineTransform to simd_float3x3 (includes ARKit's scaling + orientation)
        let displayTransform = simd_float3x3(
            simd_float3(Float(cgDisplayTransform.a), Float(cgDisplayTransform.b), 0),
            simd_float3(Float(cgDisplayTransform.c), Float(cgDisplayTransform.d), 0), 
            simd_float3(Float(cgDisplayTransform.tx), Float(cgDisplayTransform.ty), 1)
        )
        
        // Calculate aspect-fill crop scale to eliminate stretching
//        let aspectFillScale = calculateAspectFillCropScale(
//            cameraAspectRatio: cameraAspectRatio, 
//            viewportAspectRatio: viewportAspectRatio
//        )
        let aspectFillScale: Float = 1.0
        
        // Apply same scale to both X and Y to maintain camera feed aspect ratio
        let cropScale = simd_float2(aspectFillScale, aspectFillScale)
        let cropOffset = calculateCenterOffset(for: cropScale)
        
        print("🎯 ASPECT-FILL CROP: scale=\(aspectFillScale), cropScale=\(cropScale), offset=\(cropOffset)")

        let cameraTransform = CameraTransform(
            displayTransform: displayTransform,
            cropScale: cropScale,
            cropOffset: cropOffset
        )
        
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
        }
        
        let pixelFormat = CVPixelBufferGetPixelFormatType(capturedImage)
        
        if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
           pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            renderYUVFrame(capturedImage, renderEncoder: renderEncoder)
        } else {
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
        
        let yResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .r8Unorm, yWidth, yHeight, 0, &yTexture
        )
        
        let uvResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .rg8Unorm, uvWidth, uvHeight, 1, &uvTexture
        )
        
        if let yTexture = yTexture, let uvTexture = uvTexture,
           let yMetalTexture = CVMetalTextureGetTexture(yTexture),
           let uvMetalTexture = CVMetalTextureGetTexture(uvTexture) {
            renderEncoder.setFragmentTexture(yMetalTexture, index: 0)
            renderEncoder.setFragmentTexture(uvMetalTexture, index: 1)
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
