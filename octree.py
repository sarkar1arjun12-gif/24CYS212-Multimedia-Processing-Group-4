from PIL import Image
def octree_quantize_image(input_path, output_path, max_colors):
    img = Image.open(input_path).convert('RGB')
    print(f"Quantizing image to max {max_colors} colors using FASTOCTREE method...")
    quantized_img = img.quantize(colors=max_colors, method=Image.Quantize.FASTOCTREE)
    quantized_img.save(output_path)
    print(f"Quantized image saved to: {output_path}")
max_colors = int(input("Enter maximum number of colors (e.g., 256): "))
octree_quantize_image("nature.jpg","octree_output.png", max_colors)