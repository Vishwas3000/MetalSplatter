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
    
    
    // Zoom configuration - unlimited zoom range
    private let zoomStep: Float = 0.8      // Zoom increment per step
    
    // Orientation tracking
    private var currentInterfaceOrientation: UIInterfaceOrientation = .portrait
    private var currentViewportSize: CGSize = CGSize(width: 390, height: 844)  // Reasonable iPhone default
    
    
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
        // Only enable features we actually use to avoid unused texture warnings
        configuration.planeDetection = []  // Disable if not using plane detection
        configuration.environmentTexturing = .none  // Disable if not using environment lighting
        
        // Disable scene reconstruction to prevent unused semantics/confidence textures
        // if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
        //     configuration.sceneReconstruction = .mesh
        // }
        
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
            let fileExtension = url.pathExtension.lowercased()
            
            if fileExtension == "spz" {
                // Handle SPZ files
                let points = try await SPZSceneReader.read(from: url)
                try coreSplatRenderer.add(points)
            } else {
                // Handle PLY/SPLAT files (existing code)
                try await coreSplatRenderer.read(from: url)
            }
            
            print("✅ Splat loading completed successfully")
        } catch {
            print("❌ CRITICAL: Splat loading FAILED: \(error)")
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
        } else if isAREnabled {
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
        
        // No depth buffer for single-stage AR pipeline
        
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
        
        // OPTIMIZED TWO-PASS: Camera background + splats with performance optimizations
        
        // PASS 1: Render camera background - optimized for GPU debugger
        let cameraPassDescriptor = MTLRenderPassDescriptor()
        cameraPassDescriptor.colorAttachments[0].texture = colorTexture
        cameraPassDescriptor.colorAttachments[0].loadAction = .clear
        cameraPassDescriptor.colorAttachments[0].storeAction = .store
        cameraPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        // No depth buffer used for single-stage AR pipeline
        
        // Add performance optimizations
        commandBuffer.addCompletedHandler { _ in
            // Cleanup completion handler to prevent warnings
        }
        
        guard let cameraEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: cameraPassDescriptor) else {
            throw NSError(domain: "ARSplatRenderer", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create camera render encoder"])
        }
        
        cameraEncoder.label = "AR Camera Background"
        cameraEncoder.pushDebugGroup("Camera Rendering")
        
        let renderTargetSize = CGSize(width: colorTexture.width, height: colorTexture.height)
        arCameraRenderer.render(
            frame: frame,
            viewportSize: renderTargetSize,
            interfaceOrientation: getCurrentInterfaceOrientation(),
            to: cameraEncoder
        )
        
        cameraEncoder.popDebugGroup()
        cameraEncoder.endEncoding()
        
        // PASS 2: Render splats over camera background
        let arViewport = createARViewportCentered(from: frame, colorTexture: colorTexture)
        
        // Configure for optimized blending
        coreSplatRenderer.preserveExistingContent = true
        
        try coreSplatRenderer.render(
            viewports: [arViewport],
            colorTexture: colorTexture,
            colorStoreAction: colorStoreAction,
            depthTexture: nil,  // No depth buffer for AR
            rasterizationRateMap: rasterizationRateMap,
            renderTargetArrayLength: renderTargetArrayLength,
            to: commandBuffer
        )
    }
    
    
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
        
        let projectionMatrix = camera.projectionMatrix(
            for: orientation,
            viewportSize: textureSize,
            zNear: 0.01, zFar: 100.0
        )
        
        // Create model transform for splat positioning - CENTERED AT SCREEN CENTER
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
    
    // MARK: - Zoom Controls
    
    /// Zoom in the splat model
    public func zoomIn() {
        let newScale = splatScale + zoomStep
        splatScale = newScale
        Self.log.info("Zoomed in to scale: \(self.splatScale)")
    }
    
    /// Zoom out the splat model
    public func zoomOut() {
        let newScale = splatScale - zoomStep
        // Prevent negative scale values
        if newScale > 0 {
            splatScale = newScale
            Self.log.info("Zoomed out to scale: \(self.splatScale)")
        }
    }
    
    /// Set specific zoom level
    /// - Parameter scale: The scale factor (unlimited range, must be positive)
    public func setZoom(scale: Float) {
        // Only prevent negative scale values
        if scale > 0 {
            splatScale = scale
            Self.log.info("Set zoom scale to: \(self.splatScale)")
        }
    }
    
    /// Get current zoom scale value (unlimited range)
    public var zoomScale: Float {
        return splatScale
    }
    
    /// Set zoom using a multiplier of the base scale (0.1)
    /// - Parameter multiplier: Scale multiplier (e.g., 1.0 = base scale, 2.0 = double size)
    public func setZoomMultiplier(_ multiplier: Float) {
        if multiplier > 0 {
            let baseScale: Float = 0.1  // Original base scale
            splatScale = baseScale * multiplier
            Self.log.info("Set zoom multiplier to \(multiplier)x (scale: \(self.splatScale))")
        }
    }
    
    
    /// Always can zoom in (unlimited)
    public var canZoomIn: Bool {
        return true
    }
    
    /// Can zoom out if scale is positive
    public var canZoomOut: Bool {
        return splatScale > 0.001  // Prevent extremely small scales
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
//        print("*** NEW CAMERA FRAME arrived on thread: \(Thread.current)")
//        print("*** Frame timestamp: \(frame.timestamp)")
//        print("*** Frame camera intrinsics: \(frame.camera.intrinsics)")
//        print("*** Captured image size: \(CVPixelBufferGetWidth(frame.capturedImage))x\(CVPixelBufferGetHeight(frame.capturedImage))")
        
        // CRITICAL: Check if currentFrame is being updated
        if let currentFrame = session.currentFrame {
//            print("*** Session.currentFrame timestamp: \(currentFrame.timestamp)")
        } else {
//            print("*** Session.currentFrame is NIL!")
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
