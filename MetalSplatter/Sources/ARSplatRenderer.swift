#if os(iOS)

import Foundation
import Metal
import MetalKit
import ARKit
import AVFoundation
import MetalSplatter
import SampleBoxRenderer
import SplatIO
import simd
import os
import UIKit

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
    public var isAREnabled: Bool = false
    private var _isARSessionRunning: Bool = false
    
    public var isARSessionRunning: Bool {
        return _isARSessionRunning
    }
    
    // Splat positioning in AR space
    public var splatScale: Float = 0.1  // Start smaller for better visibility
    public var splatPosition: SIMD3<Float> = SIMD3(0, 0, -0.5)  // Closer to camera
    public var splatRotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    public var fixGravityFlip: Bool = true  // Apply 180° X-axis rotation to fix gravity orientation
    
    // Orientation tracking
    private var currentInterfaceOrientation: UIInterfaceOrientation = .portrait
    private var currentViewportSize: CGSize = CGSize(width: 1, height: 1)
    
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
        
        print("🔧 Creating core SplatRenderer with device: \(device.name)")
        print("   Device registry ID: \(device.registryID)")
        
        self.coreSplatRenderer = try SplatRenderer(
            device: device,
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: sampleCount,
            maxViewCount: maxViewCount,
            maxSimultaneousRenders: maxSimultaneousRenders
        )
        
        print("✅ Core SplatRenderer created successfully")
        
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
    
    deinit {
        if _isARSessionRunning {
            arSession.pause()
        }
    }
    
    // MARK: - Orientation Handling
    
    public func handleOrientationChange(_ orientation: UIInterfaceOrientation, viewportSize: CGSize) {
        currentInterfaceOrientation = orientation
        currentViewportSize = viewportSize
        let logMessage: String = "Orientation changed to: \(String(describing: orientation)), viewport: \(viewportSize)"
        Self.log.info("\(logMessage)")
    }
    
    private func getCurrentInterfaceOrientation() -> UIInterfaceOrientation {
        return currentInterfaceOrientation
    }
    
    private func getCurrentViewportSize() -> CGSize {
        return currentViewportSize
    }
    
    public func startARSession() {
        Self.log.info("startARSession called - current state: running=\(self._isARSessionRunning), enabled=\(self.isAREnabled)")
        
        guard !_isARSessionRunning else {
            Self.log.info("AR Session already running, ignoring start request")
            return
        }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }
        
        Self.log.info("Starting AR session with configuration")
        print("Starting AR session on thread: \(Thread.current)")
        
        // Check if we have camera access
        let cameraAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
        Self.log.info("Camera authorization status: \(cameraAuthorizationStatus.rawValue)")
        print("Camera auth status: \(cameraAuthorizationStatus)")
        
        arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        _isARSessionRunning = true
        isAREnabled = true
        Self.log.info("AR Session started with world tracking")
        print("AR Session started - delegate set: \(arSession.delegate != nil)")
    }
    
    public func pauseARSession() {
        guard _isARSessionRunning else {
            Self.log.info("AR Session not running, ignoring pause request")
            return
        }
        
        arSession.pause()
        _isARSessionRunning = false
        isAREnabled = false
        Self.log.info("AR Session paused")
    }
    
    // MARK: - Core SplatRenderer Interface
    
    public func reset() {
        coreSplatRenderer.reset()
    }
    
    public func read(from url: URL) async throws {
        print("🔄 ARSplatRenderer.read() called for: \(url.lastPathComponent)")
        print("   Device: \(device.name), Registry ID: \(device.registryID)")
        print("   AR Session running: \(_isARSessionRunning)")
        
        do {
            try await coreSplatRenderer.read(from: url)
            print("✅ Splat loading completed successfully")
        } catch {
            print("❌ CRITICAL: Splat loading FAILED: \(error)")
            print("   Error type: \(type(of: error))")
            print("   Error description: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("   Error domain: \(nsError.domain), code: \(nsError.code)")
            }
            throw error
        }
    }
    
    public func add(_ points: [SplatScenePoint]) throws {
        print("🔄 ARSplatRenderer.add() called with \(points.count) points")
        print("   Device: \(device.name), AR running: \(_isARSessionRunning)")
        
        do {
            try coreSplatRenderer.add(points)
            print("✅ Successfully added \(points.count) splat points")
        } catch {
            print("❌ CRITICAL: Failed to add splat points: \(error)")
            print("   Error type: \(type(of: error))")
            print("   Points count: \(points.count)")
            print("   Device memory: \(device.currentAllocatedSize / 1024 / 1024) MB")
            throw error
        }
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
        
        let hasFrame = arSession.currentFrame != nil
        let frameTimestamp = arSession.currentFrame?.timestamp ?? -1
        print("=== METAL RENDER called on thread: \(Thread.current)")
        print("=== AR enabled: \(self.isAREnabled), has frame: \(hasFrame), timestamp: \(frameTimestamp)")
        
        Self.log.info("Render called - AR enabled: \(self.isAREnabled), has frame: \(hasFrame), timestamp: \(frameTimestamp)")
        
        if isAREnabled, let currentFrame = arSession.currentFrame {
            Self.log.info("Rendering AR composition with frame timestamp: \(currentFrame.timestamp)")
            try renderARComposition(
                frame: currentFrame,
                colorTexture: colorTexture,
                colorStoreAction: colorStoreAction,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                renderTargetArrayLength: renderTargetArrayLength,
                to: commandBuffer
            )
        } else if isAREnabled {
            Self.log.info("AR enabled but no frame available - rendering fallback")
            // Render a colored background to show AR mode is active but no frames yet
            try renderFallbackBackground(
                colorTexture: colorTexture,
                colorStoreAction: colorStoreAction,
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                renderTargetArrayLength: renderTargetArrayLength,
                to: commandBuffer
            )
        } else {
            Self.log.info("Fallback to regular splat rendering")
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
    
    private func renderFallbackBackground(
        colorTexture: MTLTexture,
        colorStoreAction: MTLStoreAction,
        depthTexture: MTLTexture?,
        rasterizationRateMap: MTLRasterizationRateMap?,
        renderTargetArrayLength: Int,
        to commandBuffer: MTLCommandBuffer
    ) throws {
        Self.log.info("Rendering fallback background")
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.2, green: 0.4, blue: 0.8, alpha: 1.0)
        
        if let depthTexture = depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }
        
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw NSError(domain: "ARSplatRenderer", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create fallback render encoder"])
        }
        
        renderEncoder.label = "Fallback Background"
        renderEncoder.endEncoding()
        Self.log.info("Fallback background rendered")
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
        
        print("🎬 Starting REVERSE-ORDER AR composition (Splats → Camera)")
        
        let splatCount = coreSplatRenderer.splatBuffer.count
        
        // PASS 1: Render splats first with alpha masking
        if splatCount > 0 {
            print("✨ PASS 1: Rendering \(splatCount) splats as alpha mask")
            
            let splatRenderPassDescriptor = MTLRenderPassDescriptor()
            splatRenderPassDescriptor.colorAttachments[0].texture = colorTexture
            splatRenderPassDescriptor.colorAttachments[0].loadAction = .clear
            splatRenderPassDescriptor.colorAttachments[0].storeAction = .store
            // Clear to transparent - areas without splats will be transparent (alpha = 0)
            splatRenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            
            if let depthTexture = depthTexture {
                splatRenderPassDescriptor.depthAttachment.texture = depthTexture
                splatRenderPassDescriptor.depthAttachment.loadAction = .clear
                splatRenderPassDescriptor.depthAttachment.storeAction = .store
                splatRenderPassDescriptor.depthAttachment.clearDepth = 0.0  // Clear to near plane
            }
            
            splatRenderPassDescriptor.rasterizationRateMap = rasterizationRateMap
            splatRenderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength
            
            let arViewport = createARViewportCentered(from: frame, colorTexture: colorTexture)
            
            // Render splats with normal behavior (no preserve mode needed)
            coreSplatRenderer.preserveExistingContent = false
            
            try coreSplatRenderer.render(
                viewports: [arViewport],
                colorTexture: colorTexture,
                colorStoreAction: .store,  // Store splat results
                depthTexture: depthTexture,
                rasterizationRateMap: rasterizationRateMap,
                renderTargetArrayLength: renderTargetArrayLength,
                to: commandBuffer
            )
            
            print("   ✅ Splats rendered as alpha mask")
        } else {
            print("⚠️  No splats to render - clearing for camera background only")
            
            // No splats, just clear for camera background
            let clearRenderPassDescriptor = MTLRenderPassDescriptor()
            clearRenderPassDescriptor.colorAttachments[0].texture = colorTexture
            clearRenderPassDescriptor.colorAttachments[0].loadAction = .clear
            clearRenderPassDescriptor.colorAttachments[0].storeAction = .store
            clearRenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            
            if let depthTexture = depthTexture {
                clearRenderPassDescriptor.depthAttachment.texture = depthTexture
                clearRenderPassDescriptor.depthAttachment.loadAction = .clear
                clearRenderPassDescriptor.depthAttachment.storeAction = .store
                clearRenderPassDescriptor.depthAttachment.clearDepth = 0.0
            }
            
            guard let clearEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: clearRenderPassDescriptor) else {
                throw NSError(domain: "ARSplatRenderer", code: 3,
                             userInfo: [NSLocalizedDescriptionKey: "Failed to create clear render encoder"])
            }
            clearEncoder.label = "Clear for Camera Background"
            clearEncoder.endEncoding()
        }
        
        // PASS 2: Render camera background with depth testing
        print("📷 PASS 2: Rendering camera background with depth testing")
        
        let cameraRenderPassDescriptor = MTLRenderPassDescriptor()
        cameraRenderPassDescriptor.colorAttachments[0].texture = colorTexture
        cameraRenderPassDescriptor.colorAttachments[0].loadAction = .load  // Preserve splat pixels
        cameraRenderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        cameraRenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        if let depthTexture = depthTexture {
            cameraRenderPassDescriptor.depthAttachment.texture = depthTexture
            cameraRenderPassDescriptor.depthAttachment.loadAction = .load  // Preserve splat depths
            cameraRenderPassDescriptor.depthAttachment.storeAction = .store
        }
        
        cameraRenderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        cameraRenderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength
        
        guard let cameraEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: cameraRenderPassDescriptor) else {
            throw NSError(domain: "ARSplatRenderer", code: 4,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create camera render encoder"])
        }
        
        cameraEncoder.label = "AR Camera Background Fill"
        
        // Use tracked orientation and viewport size from Metal delegate
        let viewportSize = getCurrentViewportSize()
        let interfaceOrientation = getCurrentInterfaceOrientation()
        
        print("🎥 Rendering camera with orientation: \(String(describing: interfaceOrientation)), viewport: \(viewportSize)")
        
        // Render camera background - it will only show where depth = 1.0 (no splats rendered)
        arCameraRenderer.render(
            frame: frame,
            viewportSize: viewportSize,
            interfaceOrientation: interfaceOrientation,
            to: cameraEncoder
        )
        
        cameraEncoder.endEncoding()
        
        print("✅ Reverse-order AR composition completed")
        print("   Expected behavior:")
        print("   - Where splats exist (depth > 0.0): Splats visible, camera blocked by .equal test")
        print("   - Where no splats exist (depth = 0.0): Camera visible via .equal test")
        print("   Result: Splats mask camera feed - camera visible only where no splats exist")
    }
    
    // This method is no longer used - we're using two-pass rendering instead
    // private func renderSplatsInSinglePass(...) - REMOVED
    
    private func createARViewportCentered(from frame: ARFrame, colorTexture: MTLTexture) -> SplatRenderer.ViewportDescriptor {
        let camera = frame.camera
        let textureSize = CGSize(width: colorTexture.width, height: colorTexture.height)
        
        let viewport = MTLViewport(
            originX: 0, originY: 0,
            width: Double(textureSize.width), height: Double(textureSize.height),
            znear: 0, zfar: 1
        )
        
        // Use ARKit's projection matrix with proper orientation detection
        let orientation: UIInterfaceOrientation
        if textureSize.width > textureSize.height {
            orientation = .landscapeRight
        } else {
            orientation = .portrait
        }
        
        print("📱 Using orientation: \(orientation) for viewport: \(textureSize)")
        
        let projectionMatrix = camera.projectionMatrix(
            for: orientation,
            viewportSize: textureSize,
            zNear: 0.01, zFar: 100.0
        )
        
        // Create model transform for splat positioning - CENTERED AT SCREEN CENTER
        print("🎯 Positioning splats at screen center")
        
        // Scale down for better visibility in AR
        let scaleMatrix = matrix4x4_scale(splatScale, splatScale, splatScale)
        
        // Position splats at screen center, in front of camera
        // Z = -0.5 means 0.5 meters in front of the camera
        let centerPosition = SIMD3<Float>(0, 0, -0.5)  // Screen center, 0.5m in front
        let translationMatrix = matrix4x4_translation(centerPosition.x, centerPosition.y, centerPosition.z)
        
        // Fix gravity flip by rotating 180° around X-axis to flip Y-axis
        // This corrects the coordinate system difference between splat data and AR camera space
        let gravityFixRotation = fixGravityFlip ? 
            matrix4x4_rotation(radians: Float.pi, axis: SIMD3<Float>(1, 0, 0)) : 
            matrix4x4_identity()
        let rotationMatrix = simd_float4x4(splatRotation)
        
        let modelMatrix = translationMatrix * gravityFixRotation * rotationMatrix * scaleMatrix
        let viewMatrix = camera.viewMatrix(for: orientation) * modelMatrix
        
        print("   Splat position: \(centerPosition)")
        print("   Splat scale: \(splatScale)")
        print("   Screen size: \(textureSize)")
        
        return SplatRenderer.ViewportDescriptor(
            viewport: viewport,
            projectionMatrix: projectionMatrix,
            viewMatrix: viewMatrix,
            screenSize: SIMD2(x: Int(textureSize.width), y: Int(textureSize.height))
        )
    }
    
    public func resetSplatTransform() {
        splatScale = 0.1
        splatPosition = SIMD3(0, 0, -0.5)
        splatRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }
}

// MARK: - ARSessionDelegate

extension ARSplatRenderer: ARSessionDelegate {
    public func session(_ session: ARSession, didFailWithError error: Error) {
        Self.log.error("AR Session failed: \(error.localizedDescription)")
        print("AR Session FAILED: \(error)")
        _isARSessionRunning = false
        isAREnabled = false
    }
    
    public func sessionWasInterrupted(_ session: ARSession) {
        Self.log.info("AR Session was interrupted")
        print("AR Session INTERRUPTED")
        _isARSessionRunning = false
    }
    
    public func sessionInterruptionEnded(_ session: ARSession) {
        Self.log.info("AR Session interruption ended")
        print("AR Session interruption ENDED")
        _isARSessionRunning = true
    }
    
    public func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        switch camera.trackingState {
        case .normal:
            Self.log.info("AR Camera tracking: Normal")
            print("AR Camera tracking: NORMAL")
        case .notAvailable:
            Self.log.warning("AR Camera tracking: Not Available")
            print("AR Camera tracking: NOT AVAILABLE")
        case .limited(let reason):
            Self.log.warning("AR Camera tracking limited: \(String(describing: reason))")
            print("AR Camera tracking LIMITED: \(reason)")
        }
    }
    
    // This is critical - it's called when new frames arrive!
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        print("*** NEW CAMERA FRAME arrived on thread: \(Thread.current)")
        print("*** Frame timestamp: \(frame.timestamp)")
        print("*** Frame camera intrinsics: \(frame.camera.intrinsics)")
        print("*** Captured image size: \(CVPixelBufferGetWidth(frame.capturedImage))x\(CVPixelBufferGetHeight(frame.capturedImage))")
        
        // CRITICAL: Check if currentFrame is being updated
        if let currentFrame = session.currentFrame {
            print("*** Session.currentFrame timestamp: \(currentFrame.timestamp)")
        } else {
            print("*** Session.currentFrame is NIL!")
        }
    }
}

// MARK: - Matrix Utility Functions

fileprivate func matrix4x4_identity() -> simd_float4x4 {
    return simd_float4x4(
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(0, 0, 0, 1)
    )
}

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
