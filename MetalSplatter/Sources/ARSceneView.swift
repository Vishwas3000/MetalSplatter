import SwiftUI
import MetalKit
import SplatIO

#if canImport(UIKit) && os(iOS)
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
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    public func makeUIView(context: UIViewRepresentableContext<ARSceneView>) -> MTKView {
        let metalKitView = MTKView()
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }
        
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        do {
            let renderer = try ARSplatRenderer(
                device: metalKitView.device!,
                colorFormat: metalKitView.colorPixelFormat,
                depthFormat: metalKitView.depthStencilPixelFormat,
                sampleCount: metalKitView.sampleCount,
                maxViewCount: 1,
                maxSimultaneousRenders: 3
            )
            
            context.coordinator.renderer = renderer
            metalKitView.delegate = ARSceneViewDelegate(renderer: renderer)
            
            Task {
                await loadModel(renderer: renderer)
            }
            
            renderer.isAREnabled = isAREnabled
        } catch {
            print("Error creating AR splat renderer: \(error.localizedDescription)")
        }
        
        return metalKitView
    }
    
    public func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<ARSceneView>) {
        guard let renderer = context.coordinator.renderer else { return }
        
        renderer.isAREnabled = isAREnabled
        
        Task {
            await loadModel(renderer: renderer)
        }
    }
    
    private func loadModel(renderer: ARSplatRenderer) async {
        do {
            switch modelIdentifier {
            case .gaussianSplat(let url):
                try await renderer.read(from: url)
            case .sampleBox:
                // For now, ARSplatRenderer doesn't directly support SampleBoxRenderer
                // This would need additional integration
                break
            case .none:
                break
            }
        } catch {
            print("Error loading model: \(error.localizedDescription)")
        }
    }
}

class ARSceneViewDelegate: NSObject, MTKViewDelegate {
    private let renderer: ARSplatRenderer
    
    init(renderer: ARSplatRenderer) {
        self.renderer = renderer
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let commandBuffer = view.device?.makeCommandQueue()?.makeCommandBuffer() else { return }
        
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
        } catch {
            print("Error rendering AR scene: \(error.localizedDescription)")
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle drawable size changes if needed
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