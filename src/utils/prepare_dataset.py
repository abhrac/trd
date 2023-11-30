import os
import shutil


def parse_annots(annots_path):
    with open(annots_path, 'r') as f:
        lines = f.readlines()
        parsed = map(lambda line: line.strip().split(' '), lines)
        inst_dict = {}
        for inst_name, cls in parsed:
            if cls in inst_dict.keys():
                inst_dict[cls].append(inst_name)
            else:
                inst_dict[cls] = [inst_name]
        return inst_dict


def write2tgt(src_path, tgt_path, annots):
    for cls, inst_names in annots.items():
        cls_folder = os.path.join(tgt_path, cls)
        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder)
        for inst_name in inst_names:
            shutil.copy(os.path.join(src_path, inst_name), os.path.join(cls_folder, inst_name))


def main():
    ds_root = '/home/ac1151/Datasets/cotton_soy/data-20220324T113351Z-001/data/soyloc/'

    train_annots_path = os.path.join(ds_root, 'anno/train.txt')
    val_annots_path = os.path.join(ds_root, 'anno/val.txt')
    test_annots_path = os.path.join(ds_root, 'anno/test.txt')

    src_path = os.path.join(ds_root, 'images')
    train_tgt_path = os.path.join(ds_root, 'train')
    val_tgt_path = os.path.join(ds_root, 'val')
    test_tgt_path = os.path.join(ds_root, 'test')

    train_annots = parse_annots(train_annots_path)
    val_annots = parse_annots(val_annots_path)
    test_annots = parse_annots(test_annots_path)

    write2tgt(src_path, train_tgt_path, train_annots)
    write2tgt(src_path, val_tgt_path, val_annots)
    write2tgt(src_path, test_tgt_path, test_annots)


if __name__ == '__main__':
    main()
