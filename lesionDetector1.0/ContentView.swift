//
//  ContentView.swift
//  lesionDetector1.0
//
//  Created by Ryan Dong on 8/13/24.
//

import SwiftUI
import PhotosUI
import CoreML
import Vision

struct ContentView: View {
    @State private var selectedImage: UIImage? = nil
    @State private var isImagePickerPresented = false
    @State private var classificationLabel = ""
    
    var body: some View {
        NavigationView {
            VStack {
                if let selectedImage = selectedImage {
                    Image(uiImage: selectedImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: 300)
                        .padding()
                } else {
                    Text("No Image Selected")
                        .foregroundColor(.gray)
                        .padding()
                }

                Button(action: {
                    isImagePickerPresented = true
                }) {
                    Text("Upload Photo")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding()

                // Placeholder for custom functionality
                Button(action: {
                    classifyImage(with: selectedImage)
                }) {
                    Text("Predict")
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding()
                .disabled(selectedImage == nil) // Disable button if no image is selected
                if !classificationLabel.isEmpty {
                    Text("Classification: \(classificationLabel)")
                        .font(.title)
                        .padding()
                }
            }
            .navigationTitle("Lesion Classification")
            .sheet(isPresented: $isImagePickerPresented) {
                ImagePicker(selectedImage: $selectedImage)
            }
        }
    }
    
    // Replace with your custom functionality
    func classifyImage(with image: UIImage?) {
        guard let image = image else {return}
        
        guard let ciImage = CIImage(image: image) else {
            print("Unable to convert UIImage to CIImage")
            return
        }
        //load coreml model
        do {
            let model = try lesionClassifier_1(configuration: MLModelConfiguration()).model
            
            let request = VNCoreMLRequest(model: try VNCoreMLModel(for: model)) { request, error in
                            if let results = request.results as? [VNClassificationObservation],
                               let topResult = results.first {
                                // Update classificationLabel with the result
                                DispatchQueue.main.async {
                                    classificationLabel = topResult.identifier
                                }
                            } else {
                                print("Failed to classify image: \(String(describing: error))")
                            }
                        }
            let handler = VNImageRequestHandler(ciImage: ciImage)
            try handler.perform([request])
            
        } catch {
            print("Error loading CoreML model: \(error.localizedDescription)")
        }
        
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            
            guard let provider = results.first?.itemProvider else { return }
            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { [weak self] image, _ in
                    self?.parent.selectedImage = image as? UIImage
                }
            }
        }
    }
}

