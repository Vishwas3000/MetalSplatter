#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import SwiftUI

#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    // Multi-axis rotation control
    var rotationX: Angle = .zero  // Pitch (up/down)
    var rotationY: Angle = .zero  // Yaw (left/right)
    var rotationZ: Angle = .zero  // Roll (optional)
    
    // Rotation velocities for momentum
    var rotationVelocityX: Float = 0.0
    var rotationVelocityY: Float = 0.0
    
    // Zoom/Scale control - unlimited zoom range
    var scale: Float = 1.0
    var scaleVelocity: Float = 0.0
    
    // Gesture tracking
    var lastPanLocation: CGPoint = .zero
    var isPanning: Bool = false
    var isPinching: Bool = false
    
    // Sensitivity and momentum settings
    private let rotationSensitivity: Float = 0.008
    private let scaleSensitivity: Float = 0.01
    private let momentumDecay: Float = 0.92
    private let minimumVelocity: Float = 0.001
    private let minimumScaleVelocity: Float = 0.01

    var drawableSize: CGSize = .zero

    init?(_ metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        super.init()
        setupGestureRecognizers()
    }

    private func setupGestureRecognizers() {
#if os(iOS)
        // Pan gesture for rotation
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePanGesture(_:)))
        panGesture.maximumNumberOfTouches = 1
        metalKitView.addGestureRecognizer(panGesture)
        
        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinchGesture(_:)))
        metalKitView.addGestureRecognizer(pinchGesture)
        
        // Allow simultaneous gestures
        panGesture.delegate = self
        pinchGesture.delegate = self
        
        // Enable user interaction
        metalKitView.isUserInteractionEnabled = true
#elseif os(macOS)
        // macOS gesture handling would go here if needed
        // For now, focusing on iOS implementation
#endif
    }

#if os(iOS)
    @objc private func handlePanGesture(_ gesture: UIPanGestureRecognizer) {
        let location = gesture.location(in: metalKitView)
        
        switch gesture.state {
        case .began:
            isPanning = true
            lastPanLocation = location
            // Stop existing rotation momentum
            rotationVelocityX = 0.0
            rotationVelocityY = 0.0
            
        case .changed:
            guard isPanning else { return }
            
            let deltaX = Float(location.x - lastPanLocation.x)
            let deltaY = Float(location.y - lastPanLocation.y)
            
            // Apply rotation deltas
            // Horizontal swipe = rotation around Y axis (yaw)
            let yawDelta = deltaX * rotationSensitivity
            rotationY += Angle(radians: Double(yawDelta))
            
            // Vertical swipe = rotation around X axis (pitch)
            let pitchDelta = -deltaY * rotationSensitivity // Invert for natural feel
            rotationX += Angle(radians: Double(pitchDelta))
            
            // Clamp pitch to prevent over-rotation (optional)
            let maxPitch = Double.pi / 2.0 * 0.9 // 90% of 90 degrees
            rotationX = Angle(radians: max(-maxPitch, min(maxPitch, rotationX.radians)))
            
            // Calculate velocities for momentum
            let velocity = gesture.velocity(in: metalKitView)
            rotationVelocityY = Float(velocity.x) * rotationSensitivity * 0.01
            rotationVelocityX = -Float(velocity.y) * rotationSensitivity * 0.01
            
            lastPanLocation = location
            
        case .ended, .cancelled:
            isPanning = false
            // Velocities are already set from .changed state for momentum
            
        default:
            break
        }
    }
    
    @objc private func handlePinchGesture(_ gesture: UIPinchGestureRecognizer) {
        switch gesture.state {
        case .began:
            isPinching = true
            scaleVelocity = 0.0 // Stop existing scale momentum
            
        case .changed:
            guard isPinching else { return }
            
            // Apply scale change directly using gesture.scale (unlimited range)
            let newScale = scale * Float(gesture.scale)
            // Only prevent negative scale values
            if newScale > 0.001 {
                scale = newScale
            }
            
            // Calculate velocity for momentum
            scaleVelocity = Float(gesture.velocity) * scaleSensitivity * 0.01
            
            // Reset gesture scale to prevent accumulation
            gesture.scale = 1.0
            
        case .ended, .cancelled:
            isPinching = false
            // Velocity is already set for momentum
            
        default:
            break
        }
    }
#endif

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try await SplatRenderer(device: device,
                                                colorFormat: metalKitView.colorPixelFormat,
                                                depthFormat: metalKitView.depthStencilPixelFormat,
                                                sampleCount: metalKitView.sampleCount,
                                                maxViewCount: 1,
                                                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            try await splat.read(from: url)
            modelRenderer = splat
        case .sampleBox:
            modelRenderer = try! await SampleBoxRenderer(device: device,
                                                         colorFormat: metalKitView.colorPixelFormat,
                                                         depthFormat: metalKitView.depthStencilPixelFormat,
                                                         sampleCount: metalKitView.sampleCount,
                                                         maxViewCount: 1,
                                                         maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    private var viewport: ModelRendererViewportDescriptor {
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(Constants.fovy.radians),
                                                             aspectRatio: Float(drawableSize.width / drawableSize.height),
                                                             nearZ: 0.1,
                                                             farZ: 100.0)

        // Create rotation matrices for each axis
        let rotationMatrixX = matrix4x4_rotation(radians: Float(rotationX.radians),
                                                 axis: SIMD3<Float>(1, 0, 0)) // X-axis (pitch)
        let rotationMatrixY = matrix4x4_rotation(radians: Float(rotationY.radians),
                                                 axis: SIMD3<Float>(0, 1, 0)) // Y-axis (yaw)
        let rotationMatrixZ = matrix4x4_rotation(radians: Float(rotationZ.radians),
                                                 axis: SIMD3<Float>(0, 0, 1)) // Z-axis (roll)
        
        // Scale matrix
        let scaleMatrix = matrix4x4_scale(scale, scale, scale)
        
        // Translation matrix
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        
        // Turn common 3D GS PLY files rightside-up
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        // Combine transformations: Translation * Scale * RotationY * RotationX * RotationZ * Calibration
        let combinedMatrix = translationMatrix * scaleMatrix * rotationMatrixY * rotationMatrixX * rotationMatrixZ * commonUpCalibration

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: combinedMatrix,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    private func updateMomentum() {
        // Only apply momentum when not actively gesturing
        guard !isPanning && !isPinching else { return }
        
        // Apply rotation momentum
        if abs(rotationVelocityX) > minimumVelocity {
            rotationX += Angle(radians: Double(rotationVelocityX))
            
            // Clamp pitch with momentum
            let maxPitch = Double.pi / 2.0 * 0.9
            rotationX = Angle(radians: max(-maxPitch, min(maxPitch, rotationX.radians)))
            
            rotationVelocityX *= momentumDecay
        } else {
            rotationVelocityX = 0.0
        }
        
        if abs(rotationVelocityY) > minimumVelocity {
            rotationY += Angle(radians: Double(rotationVelocityY))
            rotationVelocityY *= momentumDecay
        } else {
            rotationVelocityY = 0.0
        }
        
        // Apply scale momentum (unlimited range)
        if abs(scaleVelocity) > minimumScaleVelocity {
            let newScale = scale + scaleVelocity
            // Only prevent negative scale values
            if newScale > 0.001 {
                scale = newScale
            }
            scaleVelocity *= momentumDecay
        } else {
            scaleVelocity = 0.0
        }
    }
    
    // Reset to default view
    func resetCamera() {
        rotationX = .zero
        rotationY = .zero
        rotationZ = .zero
        scale = 1.0
        rotationVelocityX = 0.0
        rotationVelocityY = 0.0
        scaleVelocity = 0.0
    }

    func draw(in view: MTKView) {
        guard let modelRenderer else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        // Update momentum-based transformations
        updateMomentum()

        do {
            try modelRenderer.render(viewports: [viewport],
                                     colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                     colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                     depthTexture: view.depthStencilTexture,
                                     rasterizationRateMap: nil,
                                     renderTargetArrayLength: 0,
                                     to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        commandBuffer.present(drawable)

        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }
}

// MARK: - UIGestureRecognizerDelegate
#if os(iOS)
extension MetalKitSceneRenderer: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        // Allow pan and pinch gestures to work simultaneously
        return true
    }
}
#endif

// MARK: - Matrix Helper Functions
fileprivate func matrix4x4_scale(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    return simd_float4x4(
        SIMD4<Float>(x, 0, 0, 0),
        SIMD4<Float>(0, y, 0, 0),
        SIMD4<Float>(0, 0, z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    )
}

#endif // os(iOS) || os(macOS)
