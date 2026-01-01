import json
import os
import argparse

def convert_to_llamafactory(input_path, output_path, image_folder):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []
    
    for item in data:
        # Giả sử file json của bạn có các trường: 'image', 'question', 'answer'
        # Nếu tên trường khác, hãy sửa lại ở đây (vd: item['img_path']...)
        image_path = os.path.join(image_folder, item.get('image', ''))
        
        entry = {
            "images": [image_path],
            "messages": [
                {
                    "role": "user",
                    "content": "<image>" + item.get('question', '')
                },
                {
                    "role": "assistant",
                    "content": item.get('answer', '')
                }
            ]
        }
        formatted_data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Đã convert {len(data)} mẫu sang: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input json")
    parser.add_argument("--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing images")
    args = parser.parse_args()

    convert_to_llamafactory(args.input, args.output, args.image_dir)