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
    
    private struct Vertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
    }
    
    private let quadVertices: [Vertex] = [
        Vertex(position: SIMD2(-1, -1), texCoord: SIMD2(0, 1)),
        Vertex(position: SIMD2( 1, -1), texCoord: SIMD2(1, 1)),
        Vertex(position: SIMD2(-1,  1), texCoord: SIMD2(0, 0)),
        Vertex(position: SIMD2( 1,  1), texCoord: SIMD2(1, 0))
    ]
    
    public init?(device: MTLDevice) {
        self.device = device
        
        do {
            self.library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            Self.log.error("Failed to create MetalSplatterAR library: \(error)")
            return nil
        }
        
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
            Self.log.error("Failed to create CVMetalTextureCache")
            return nil
        }
        
        setupPipelineState()
        setupVertexBuffer()
    }
    
    private func setupPipelineState() {
        guard let vertexFunction = library.makeFunction(name: "arCameraVertexShader"),
              let fragmentFunction = library.makeFunction(name: "arCameraFragmentShader") else {
            Self.log.error("Failed to load AR camera shader functions")
            return
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
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
    
    public func render(
        frame: ARFrame,
        to renderEncoder: MTLRenderCommandEncoder
    ) {
        guard let pipelineState = pipelineState,
              let vertexBuffer = vertexBuffer else {
            Self.log.error("AR camera renderer not properly initialized")
            return
        }
        
        renderEncoder.pushDebugGroup("AR Camera Background")
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        let capturedImage = frame.capturedImage
        let pixelFormat = CVPixelBufferGetPixelFormatType(capturedImage)
        
        if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
           pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            renderYUVFrame(capturedImage, renderEncoder: renderEncoder)
        } else {
            renderRGBFrame(capturedImage, renderEncoder: renderEncoder)
        }
        
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.popDebugGroup()
    }
    
    private func renderYUVFrame(_ pixelBuffer: CVPixelBuffer, renderEncoder: MTLRenderCommandEncoder) {
        var yTexture: CVMetalTexture?
        var uvTexture: CVMetalTexture?
        
        let yWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let uvWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uvHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, textureCache, pixelBuffer, nil,
            .r8Unorm, yWidth, yHeight, 0, &yTexture
        )
        
        CVMetalTextureCacheCreateTextureFromImage(
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