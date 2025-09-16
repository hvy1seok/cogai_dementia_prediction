import os

def generate_metadata(root_path='../Pitt/textdata/', output_path='../../Pitt/'):
    control_root = os.path.join(root_path, 'Control')
    dementia_root = os.path.join(root_path, 'Dementia')

    control_meta = []
    dementia_meta = []

    tasks = ['cookie', 'fluency', 'recall', 'sentence']  # 소문자 주의

    for task in tasks:
        control_path = os.path.join(control_root, task)
        dementia_path = os.path.join(dementia_root, task)

        # Control group
        if os.path.exists(control_path):
            for file in os.listdir(control_path):
                if file.endswith('.cha'):
                    rel_path = os.path.join('Control', task, file)
                    control_meta.append(rel_path)

        # Dementia group
        if os.path.exists(dementia_path):
            for file in os.listdir(dementia_path):
                if file.endswith('.cha'):
                    rel_path = os.path.join('Dementia', task, file)
                    dementia_meta.append(rel_path)

    # Write metadata files to preprocess directory
    with open(os.path.join(output_path, 'controlCha.txt'), 'w') as f:
        for line in sorted(control_meta):
            f.write(line + '\n')

    with open(os.path.join(output_path, 'dementiaCha.txt'), 'w') as f:
        for line in sorted(dementia_meta):
            f.write(line + '\n')

    print("✅ 메타데이터 생성 완료")
    print(f" - controlCha.txt: {len(control_meta)}개")
    print(f" - dementiaCha.txt: {len(dementia_meta)}개")

if __name__ == "__main__":
    generate_metadata('../Pitt/textdata/', './')  # output은 preprocess/ 폴더로
