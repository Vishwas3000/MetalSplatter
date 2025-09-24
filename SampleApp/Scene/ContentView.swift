import SwiftUI
import RealityKit
import UniformTypeIdentifiers
import MetalSplatter

struct ContentView: View {
    @State private var isPickingFile = false
    @State private var isARModeEnabled = true

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(macOS) || os(visionOS)
        mainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            mainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    VStack {
                        #if os(iOS)
                        HStack {
                            Text("AR Mode")
                            Toggle("", isOn: $isARModeEnabled)
                                .labelsHidden()
                        }
                        .padding()
                        #endif
                        
                        #if os(iOS)
                        if isARModeEnabled {
                            ARSceneView(modelIdentifier: ARModelIdentifier(from: modelIdentifier), isAREnabled: $isARModeEnabled)
                                .navigationTitle(modelIdentifier.description + " (AR)")
                        } else {
                            MetalKitSceneView(modelIdentifier: modelIdentifier)
                                .navigationTitle(modelIdentifier.description)
                        }
                        #else
                        MetalKitSceneView(modelIdentifier: modelIdentifier)
                            .navigationTitle(modelIdentifier.description)
                        #endif
                    }
                }
        }
#endif // os(iOS)
    }

    @ViewBuilder
    var mainView: some View {
        VStack {
            Spacer()

            Text("MetalSplatter SampleApp")

            Spacer()

            Button("Read Scene File") {
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif
            .fileImporter(isPresented: $isPickingFile,
                          allowedContentTypes: [
                            UTType(filenameExtension: "ply")!,
                            UTType(filenameExtension: "splat")!,
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    _ = url.startAccessingSecurityScopedResource()
                    Task {
                        // This is a sample app. In a real app, this should be more tightly scoped, not using a silly timer.
                        try await Task.sleep(for: .seconds(10))
                        url.stopAccessingSecurityScopedResource()
                    }
                    openWindow(value: ModelIdentifier.gaussianSplat(url))
                case .failure:
                    break
                }
            }

            Spacer()

            Button("Show Sample Box") {
                openWindow(value: ModelIdentifier.sampleBox)
            }
            .padding()
            .buttonStyle(.borderedProminent)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif

            Spacer()

#if os(visionOS)
            Button("Dismiss Immersive Space") {
                Task {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
            .disabled(!immersiveSpaceIsShown)

            Spacer()
#endif // os(visionOS)
        }
    }
}
