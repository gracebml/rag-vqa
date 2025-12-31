import json
import re
import os

input_path = 'data/knowledge_base.json'
output_path = 'data/knowledge_base_clean.json'

blacklist_entity = [
    "nhà xuất bản", "nxb", "công ty", "phát hành", 
    "bìa mềm", "bìa cứng", "tái bản", "trang", "khổ sách", 
    "giá bìa", "isbn", "mã sách", "lời nói đầu", "mục lục",
    "ban biên soạn", "hội đồng", "chỉ đạo tổ chức", "viện hàn lâm khoa học xã hội việt nam"
]

author_keywords = [
    "chủ biên", "đồng chủ biên", "tổng chủ biên",
    "tham gia biên soạn", "người biên soạn", "thư ký khoa học",
    "pgs.ts", "ts.", "thạc sĩ", "tiến sĩ", "nghiên cứu viên", "ncvc",
    "bản quyền", "chịu trách nhiệm", "Lịch sử Việt Nam Tập", "bộ sách Lịch sử Việt Nam",
    "bộ sách lịch sử việt nam.", "xuất bản năm", "Viện Hàn lâm Khoa học Xã hội Việt Nam"
]

def is_valid_entity(item):
    entity = str(item.get('entity', '').lower())
    facts = str(item.get('facts', ''))
    summary = str(item.get('summary', ''))

    content = facts + " " + summary
    for word in blacklist_entity:
        if word in entity:
            return False
    
    for word in author_keywords:
        if word in content:
            return False
    
    if len(facts) < 10 and len(summary) < 10:
        return False
    
    return True

def clean_text(text):
    if not text:
        return
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return ""

    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    try: 
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Original size: {len(data)} entities")
        cleaned_data = []
        removed_count = 0
        for item in data:
            if is_valid_entity(item):
                item['entity'] = clean_text(item.get('entity', ''))
                item['facts'] = clean_text(item.get('facts', ''))
                item['summary'] = clean_text(item.get('summary', ''))
                if item['facts'] or item['summary']:
                    cleaned_data.append(item)
            else:
                removed_count = removed_count + 1
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent = 2, ensure_ascii=False)
        
        print(f"Đã lọc bỏ: {removed_count} entities (NXB, Tác giả, Rác...)")
        print(f"Giữ lại:   {len(cleaned_data)} entities")
        print(f"Saved to:  {output_path}")

    except FileExistsError:
        print(f"Error: Not found file {input_path}")
    except Exception as e:
        print(f"Error: {e}")