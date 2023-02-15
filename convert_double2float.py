import xml.etree.ElementTree as ET


def convert_double2float(source, target=None):
    # 解析PMML文件
    tree = ET.parse(source)
    root = tree.getroot()

    # 遍历Double元素，并将其值转换为Float类型
    for double_element in root.iter('{http://www.dmg.org/PMML-4_4}Double'):
        double_element.text = str(float(double_element.text))

    # 将修改后的PMML文件保存回磁盘
    tree.write(target if target else source)


if __name__ == '__main__':
    source = "scorecard.pmml"
    convert_double2float(source, target="scorecard_float.pmml")
