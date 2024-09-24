import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torchvision.transforms as transforms
import torch
from torchvision import models

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.br = CvBridge()
        
        # 设置 DeepLabV3 模型
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()  # 设置为评估模式
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),  # DeepLabV3 输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 打开 USB 摄像头
        self.cap = cv2.VideoCapture(0)

        # 设置图像发布者
        self.publisher = self.create_publisher(Image, '/image_processed', 10)

    def process_frame(self, frame):
        # 将图像转换为 PyTorch Tensor
        tensor_frame = self.transform(frame)
        tensor_frame = tensor_frame.unsqueeze(0)  # 添加 batch 维度

        # 使用模型进行推理
        with torch.no_grad():
            output = self.model(tensor_frame)['out'][0]  # 获取输出
            output_predictions = output.argmax(0)  # 获取每个像素的预测类别

        return output_predictions.cpu().numpy()  # 返回 NumPy 数组

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error('Failed to capture image')
                break
            
            # 处理图像
            output = self.process_frame(frame)

            # 将输出数组转换为图像（可视化）
            output_image = (output * 255 / output.max()).astype('uint8')  # 归一化到 0-255
            output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_JET)  # 应用伪彩色

            # 将图像转换为 ROS 消息
            ros_image = self.br.cv2_to_imgmsg(output_image, encoding="bgr8")
            self.publisher.publish(ros_image)

            cv2.imshow("USB Camera", frame)
            cv2.imshow("Segmentation Output", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    image_processor.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()