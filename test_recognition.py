from recognize import recognize_face

# Path to the image you want to test
test_image = "test_photo.jpg"  # Replace with your image path

result = recognize_face(test_image)
print("Recognition Result:", result)