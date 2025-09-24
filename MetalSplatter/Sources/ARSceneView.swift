import SwiftUI
import MetalKit
import SplatIO

#if os(iOS)
// Note: Don't import MetalSplatter here - this file IS part of MetalSplatter!

public struct ARSceneView: UIViewRepresentable {
    public var modelIdentifier: ARModelIdentifier?
    @Binding public var isAREnabled: Bool
    
    public init(modelIdentifier: ARModelIdentifier?, isAREnabled: Binding<Bool>) {
        self.modelIdentifier = modelIdentifier
        self._isAREnabled = isAREnabled
    }
    
    public class Coordinator {
        var renderer: ARSplatRenderer?
        var isViewActive = false
        var displayLink: CADisplayLink?
        weak var metalKitView: MTKView?
        var mtkViewDelegate: ARSceneViewDelegate? // Keep strong reference to delegate
        var currentInterfaceOrientation: UIInterfaceOrientation = .portrait
        
        deinit {
            renderer?.pauseARSession()
            displayLink?.invalidate()
        }
        
        @objc func displayLinkCallback() {
            print("🔄 Display link callback triggered")
            guard let metalKitView = metalKitView,
                  isViewActive else { 
                print("❌ Display link guard failed - view: \(metalKitView != nil), active: \(isViewActive)")
                return 
            }
            
            // Try both approaches
            print("📱 Calling setNeedsDisplay on MTKView")
            metalKitView.setNeedsDisplay()
            
            // Also try calling draw() directly since setNeedsDisplay isn't working
            print("🎯 Calling draw() directly on MTKView")
            metalKitView.draw()
            
            // Check MTKView state every few frames
            if Int.random(in: 0..<60) == 0 { // Every ~1 second
                print("📊 MTKView State Check:")
                print("   Frame: \(metalKitView.frame)")
                print("   Bounds: \(metalKitView.bounds)")
                print("   Superview: \(metalKitView.superview != nil)")
                print("   Window: \(metalKitView.window != nil)")
                print("   Hidden: \(metalKitView.isHidden)")
                print("   Alpha: \(metalKitView.alpha)")
                print("   color value: \(metalKitView.backgroundColor)")
                print("   isPaused: \(metalKitView.isPaused)")
                print("   enableSetNeedsDisplay: \(metalKitView.enableSetNeedsDisplay)")
                print("   Delegate: \(metalKitView.delegate != nil)")
                print("   Device: \(metalKitView.device != nil)")
            }
        }
        
        func startDisplayLink() {
            guard displayLink == nil else { return }
            displayLink = CADisplayLink(target: self, selector: #selector(displayLinkCallback))
            displayLink?.add(to: .main, forMode: .common)
            print("Display link started")
        }
        
        func stopDisplayLink() {
            displayLink?.invalidate()
            displayLink = nil
            print("Display link stopped")
        }
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    public func makeUIView(context: UIViewRepresentableContext<ARSceneView>) -> MTKView {
        let metalKitView = MTKView()
        
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            print("Failed to create Metal device")
            return metalKitView
        }
        
        // First, create the renderer and delegate BEFORE configuring MTKView
        do {
            print("Creating ARSplatRenderer with device: \(metalDevice.name)")
            let renderer = try ARSplatRenderer(
                device: metalDevice,
                colorFormat: MTLPixelFormat.bgra8Unorm_srgb,
                depthFormat: MTLPixelFormat.depth32Float,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: 3
            )
            
            print("ARSplatRenderer created successfully")
            
            context.coordinator.renderer = renderer
            context.coordinator.isViewActive = true
            context.coordinator.metalKitView = metalKitView
            
            let delegate = ARSceneViewDelegate(renderer: renderer, coordinator: context.coordinator)
            
            // Keep a strong reference to the delegate in coordinator FIRST
            context.coordinator.mtkViewDelegate = delegate
            
            // NOW configure the MTKView with delegate set
            metalKitView.device = metalDevice
            metalKitView.delegate = delegate  // SET DELEGATE EARLY
            metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
            metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
            metalKitView.sampleCount = 1
            metalKitView.clearColor = MTLClearColor(red: 0.1, green: 0.5, blue: 0, alpha: 0.5)
            metalKitView.backgroundColor = .cyan
            
            // Configure rendering mode AFTER delegate is set
            print("🔧 Configuring MTKView for AUTOMATIC rendering mode...")
            metalKitView.enableSetNeedsDisplay = false  // Automatic mode
            metalKitView.isPaused = false               // Not paused  
            metalKitView.preferredFramesPerSecond = 60  // 60 FPS
            
            print("   enableSetNeedsDisplay: \(metalKitView.enableSetNeedsDisplay)")
            print("   isPaused: \(metalKitView.isPaused)")
            print("   preferredFramesPerSecond: \(metalKitView.preferredFramesPerSecond)")
            print("   delegate set: \(metalKitView.delegate != nil)")
            
            print("MTKView configured - device: \(metalDevice.name)")
            
            // Explicitly trigger drawing
            metalKitView.setNeedsDisplay()
            print("Triggered initial setNeedsDisplay")
            
            // Start the manual display link
            context.coordinator.startDisplayLink()
            
            // Start AR session first if needed
            if isAREnabled {
                print("Starting AR session immediately")
                renderer.startARSession()
                
                // Load model AFTER AR session starts and MTKView is set up
                Task {
                    // Wait for AR session to initialize properly
                    try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
                    
                    // Additional check: ensure MTKView has proper bounds before loading
                    await MainActor.run {
                        print("🎯 Pre-loading checks:")
                        print("   MTKView frame: \(metalKitView.frame)")
                        print("   MTKView bounds: \(metalKitView.bounds)")
                        print("   MTKView window: \(metalKitView.window != nil)")
                        print("   AR session running: \(renderer.isARSessionRunning)")
                    }
                    
                    if metalKitView.bounds.width > 0 && metalKitView.bounds.height > 0 {
                        print("✅ MTKView ready, loading model...")
                        await loadModel(renderer: renderer)
                    } else {
                        print("⚠️  MTKView not ready yet, delaying model load...")
                        try await Task.sleep(nanoseconds: 1_000_000_000) // Wait 1 more second
                        await loadModel(renderer: renderer)
                    }
                }
            } else {
                print("AR not enabled, loading model immediately")
                Task {
                    await loadModel(renderer: renderer)
                }
            }
            
            // Force the MTKView to start drawing and check state
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                print("🔍 Initial MTKView state check after setup:")
                print("   Frame: \(metalKitView.frame)")
                print("   Bounds: \(metalKitView.bounds)")
                print("   isPaused: \(metalKitView.isPaused)")
                print("   enableSetNeedsDisplay: \(metalKitView.enableSetNeedsDisplay)")
                print("   preferredFramesPerSecond: \(metalKitView.preferredFramesPerSecond)")
                print("   delegate set: \(metalKitView.delegate != nil)")
                print("   superview: \(metalKitView.superview != nil)")
                print("   window: \(metalKitView.window != nil)")
                print("   isHidden: \(metalKitView.isHidden)")
                print("   alpha: \(metalKitView.alpha)")
                
                print("🚀 Manually calling setNeedsDisplay")
                metalKitView.setNeedsDisplay()
                
                // Also try draw directly
                print("🚀 Manually calling draw")
                metalKitView.draw()
            }
        } catch {
            print("FATAL ERROR creating AR splat renderer: \(error)")
            print("Error details: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("Error domain: \(nsError.domain), code: \(nsError.code)")
                print("Error user info: \(nsError.userInfo)")
            }
        }
        
        print("📦 Returning MTKView from makeUIView")
        print("   Initial frame: \(metalKitView.frame)")
        print("   Initial bounds: \(metalKitView.bounds)")
        
        // Ensure MTKView gets proper autoresizing
        metalKitView.translatesAutoresizingMaskIntoConstraints = true
        metalKitView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        return metalKitView
    }
    
    public func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<ARSceneView>) {
        print("🔄 SwiftUI updateUIView called")
        print("   View frame: \(view.frame)")
        print("   AR enabled: \(isAREnabled)")
        
        guard let renderer = context.coordinator.renderer else { 
            print("❌ No renderer in updateUIView")
            return 
        }
        
        // Handle AR session state changes
        if isAREnabled && !renderer.isARSessionRunning {
            print("🚀 Starting AR session from updateUIView")
            renderer.startARSession()
        } else if !isAREnabled && renderer.isARSessionRunning {
            print("⏸️ Pausing AR session from updateUIView")
            renderer.pauseARSession()
        }
        
        // Force redraw
        print("🚀 Forcing setNeedsDisplay from updateUIView")
        view.setNeedsDisplay()
        
        Task {
            await loadModel(renderer: renderer)
        }
    }
    
    static public func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.isViewActive = false
        coordinator.stopDisplayLink()
        coordinator.renderer?.pauseARSession()
    }
    
    private func loadModel(renderer: ARSplatRenderer) async {
        print("🎯 loadModel() called with modelIdentifier: \(String(describing: modelIdentifier))")
        
        do {
            // Check GPU memory before loading large splat files in AR mode
            if renderer.isAREnabled {
                let device = renderer.device
                print("🔍 GPU Memory check for AR mode:")
                print("   Device: \(device.name)")
                print("   Recommended working set: \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
                print("   Current allocated: \(device.currentAllocatedSize / 1024 / 1024) MB")
                
                // Basic memory pressure check
                let availableMemory = Int(device.recommendedMaxWorkingSetSize) - device.currentAllocatedSize
                let estimatedSplatMemory: Int64 = 50_000_000 // ~50MB rough estimate for large splats
                
                if availableMemory < estimatedSplatMemory {
                    print("⚠️  WARNING: Low GPU memory for AR + Splats")
                    print("   Available: \(availableMemory / 1024 / 1024) MB")
                    print("   Estimated needed: \(estimatedSplatMemory / 1024 / 1024) MB")
                }
            }
            
            switch modelIdentifier {
            case .gaussianSplat(let url):
                print("📂 Loading Gaussian Splat from: \(url.lastPathComponent)")
                
                // Try loading the splat file
                do {
                    try await renderer.read(from: url)
                    print("✅ Gaussian Splat loaded successfully in AR mode")
                } catch {
                    print("💥 Splat loading FAILED in AR mode: \(error)")
                    
                    // If we're in AR mode and splat loading fails, continue without splats
                    if renderer.isAREnabled {
                        print("🔄 Continuing AR mode without splats due to loading failure")
                        print("   AR camera feed should still work")
                        // Don't rethrow - let AR mode continue without splats
                    } else {
                        // In non-AR mode, propagate the error
                        throw error
                    }
                }
                
            case .sampleBox:
                print("📦 Loading Sample Box")
                // For now, ARSplatRenderer doesn't directly support SampleBoxRenderer
                // This would need additional integration
                break
            case .none:
                print("❌ No model to load")
                break
            }
        } catch {
            print("❌ CRITICAL ERROR loading model: \(error)")
            print("   Error type: \(type(of: error))")
            print("   Error description: \(error.localizedDescription)")
            
            // For debugging - print stack trace if possible
            if let nsError = error as NSError? {
                print("   Error domain: \(nsError.domain)")
                print("   Error code: \(nsError.code)")
                print("   User info: \(nsError.userInfo)")
            }
        }
    }
}

class ARSceneViewDelegate: NSObject, MTKViewDelegate {
    private let renderer: ARSplatRenderer
    private weak var coordinator: ARSceneView.Coordinator?
    
    init(renderer: ARSplatRenderer, coordinator: ARSceneView.Coordinator) {
        self.renderer = renderer
        self.coordinator = coordinator
        super.init()
        print("ARSceneViewDelegate initialized with Metal-based orientation tracking")
    }
    
    func draw(in view: MTKView) {
        print(">>> MTKView draw called on thread: \(Thread.current)")
        print(">>> Draw cycle starting...")
        
        guard let drawable = view.currentDrawable else {
            print(">>> No drawable available")
            return
        }
        print(">>> Drawable acquired: \(drawable.texture.width)x\(drawable.texture.height)")
        
        guard let device = view.device else {
            print("No Metal device available")
            return
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Failed to create command buffer")
            return
        }
        
        print("About to render with AR enabled: \(renderer.isAREnabled)")
        
        do {
            try renderer.render(
                viewports: [], // ARSplatRenderer handles viewport creation internally
                colorTexture: view.multisampleColorTexture ?? drawable.texture,
                colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
            print("Render completed successfully")
        } catch {
            print("Error rendering AR scene: \(error)")
            print("Error details: \(error.localizedDescription)")
        }
        
        commandBuffer.present(drawable)
        
        commandBuffer.addCompletedHandler { commandBuffer in
            if let error = commandBuffer.error {
                print("Command buffer completed with error: \(error)")
            } else {
                print("Command buffer completed successfully")
            }
        }
        
        commandBuffer.commit()
        print("Frame presented and committed")
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("📐 MTKView drawableSizeWillChange called: \(size)")
        print("   View frame: \(view.frame)")
        print("   View bounds: \(view.bounds)")
        
        // Detect orientation based on aspect ratio change
        let aspectRatio = size.width / size.height
        let newOrientation: UIInterfaceOrientation
        
        if aspectRatio > 1.0 {
            // Landscape - determine which direction
            let deviceOrientation = UIDevice.current.orientation
            if deviceOrientation == .landscapeLeft {
                newOrientation = .landscapeRight  // Device left = interface right
            } else {
                newOrientation = .landscapeLeft   // Default landscape
            }
        } else {
            // Portrait - determine which direction  
            let deviceOrientation = UIDevice.current.orientation
            if deviceOrientation == .portraitUpsideDown {
                newOrientation = .portraitUpsideDown
            } else {
                newOrientation = .portrait  // Default portrait
            }
        }
        
        // Update orientation in coordinator
        if let coordinator = coordinator {
            coordinator.currentInterfaceOrientation = newOrientation
            print("🔄 Orientation detected via MTKView: \(String(describing: newOrientation)) (aspect: \(String(format: "%.2f", aspectRatio)))")
        }
        
        // Notify renderer about the orientation change
        renderer.handleOrientationChange(newOrientation, viewportSize: size)
        
        // Trigger a redraw
        view.setNeedsDisplay()
    }
}

#else

// Stub for non-iOS platforms
public struct ARSceneView: View {
    public var modelIdentifier: ARModelIdentifier?
    @Binding public var isAREnabled: Bool
    
    public init(modelIdentifier: ARModelIdentifier?, isAREnabled: Binding<Bool>) {
        self.modelIdentifier = modelIdentifier
        self._isAREnabled = isAREnabled
    }
    
    public var body: some View {
        Text("AR is only available on iOS")
            .foregroundColor(.secondary)
    }
}

#endif

// ModelIdentifier available on all platforms
public enum ARModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL)
    case sampleBox
    
    public var description: String {
        switch self {
        case .gaussianSplat(let url):
            "Gaussian Splat: \(url.path)"
        case .sampleBox:
            "Sample Box"
        }
    }
    
    // Conversion initializer from SampleApp's ModelIdentifier
    public init(from appModelId: Any) {
        // We'll use runtime introspection to handle this conversion
        let mirror = Mirror(reflecting: appModelId)
        
        if let caseName = mirror.children.first?.label {
            switch caseName {
            case "gaussianSplat":
                if let url = mirror.children.first?.value as? URL {
                    self = .gaussianSplat(url)
                    return
                }
            case "sampleBox":
                self = .sampleBox
                return
            default:
                break
            }
        }
        
        // Fallback
        self = .sampleBox
    }
}
