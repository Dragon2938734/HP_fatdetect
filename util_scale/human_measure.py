import pywavefront
import numpy as np
import os

from measurement import Body3D


def measure_body(smpla_path):
    person = pywavefront.Wavefront(
        # os.path.join(smpla_path, 'xkka2.obj'),
        smpla_path,
        create_materials=True,
        collect_faces=True
    )
    # materials = person.materials
    # for material in materials.values():
    #     import ipdb
    #     ipdb.set_trace()
    #     material.texture = None

    faces = np.array(person.mesh_list[0].faces)
    vertices = np.array(person.vertices)

    body = Body3D(vertices, faces)

    body_measurements = body.getMeasurements()

    height = body.height()
    weight = body.weight()
    shoulder_2d, shoulder_location, shoulder_length = body.shoulder()
    chest_2d, chest_location, chest_length = body.chest()
    hip_2d, hip_location, hip_length = body.hip()
    waist_2d, waist_location, waist_length = body.waist()
    thigh_2d, thigh_location, thigh_length = body.thighOutline()
    outer_leg_length = body.outerLeg()
    inner_leg_length = body.innerLeg()
    neck_2d, neck_location, neck_length = body.neck()
    neck_hip_length = body.neckToHip()

    # print('身高：',height)
    # print('体重：',weight)
    # print('肩维：',shoulder_length)
    # print('胸围：',chest_length)
    # print('臀围：',hip_length)
    # print('腰围：',waist_length)
    # print('大腿维度',thigh_length)
    # print('外腿长：',outer_leg_length)
    # print('内腿长：',inner_leg_length)
    # print('脖子维度：',neck_length)
    # print('臀到脖子：',neck_hip_length)
    # print('身高：',height)
    extraInfo = {
        "extraInfo": [{"title": '身高', 'value': height},
                      {"title": '体重', 'value': weight},
                      {"title": '脖围', 'value': neck_length},
                      {"title": '肩围', 'value': shoulder_length},
                      {"title": '胸围', 'value': chest_length},
                      {"title": '臀围', 'value': hip_length},
                      {"title": '腰围', 'value': waist_length},
                      {"title": '大腿围', 'value': thigh_length},
                      {"title": '腿长', 'value': inner_leg_length}, 
                      {"title": '下半身长', 'value': outer_leg_length},
                      {"title": '上半身长', 'value': neck_hip_length}, ]
    }
    print("extraInfo身体测量结果：", extraInfo)
    return extraInfo


if __name__ == '__main__':

    # # 读取OBJ文件
    # scene = pywavefront.Wavefront('1643143575739080705_fusion_smpl.obj')

    # materials = scene.materials
    # for material in materials.values():
    #     material.texture = None

    # # # 设置模型颜色
    # # color = (200, 200, 200)  # 灰色
    # # for v in scene.vertices:
    # #         v.color = color

    # # 保存为新的OBJ文件
    # writer = pywavefront.ObjWriter('example_white.obj', scene)
    # writer.write()

    measure_body = measure_body(smpla_path='222222_fusion_smpl.obj')


