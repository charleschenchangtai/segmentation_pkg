import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torchvision.transforms as transforms
import torch
from torchvision import models
import threading

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

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        # threading.Thread 创建了一个新的线程，并指定 self.capture_frames 作为目标函数。这个线程会在后台运行，持续尝试捕获摄像头图像。
        threading.Thread(target=self.capture_frames, daemon=True).start()
        

    # threading 1
    def capture_frames(self):        
        while rclpy.ok():
            #ret：这是一个布尔值，表示图像是否成功读取。如果成功读取，ret 的值为 True. 如果未能读取,则为 False
            #frame:这是捕获到的图像（一个数组）。如果读取成功，frame 中就包含了摄像头拍摄的那一帧图像；如果读取失败，frame 可能会是 None。
            ret, frame = self.cap.read()
            if ret:
                # 将图像缩小，减少处理负担
                frame = cv2.resize(frame, (320, 240))  # 调整尺寸
                # 获取锁：当程序执行到 with self.frame_lock: 时，线程会尝试获取锁。如果锁当前未被其他线程占用，线程将成功获取锁，并进入 with 块内部。
                # 执行代码：在 with 块内部，线程执行 self.latest_frame = frame，安全地更新了最新捕获的帧。
                # 释放锁：当程序退出 with 块时（无论是正常结束还是出现异常），Python 会自动释放锁。
                with self.frame_lock:  
                    self.latest_frame = frame  
    
    def process_frame(self, frame):
        # 将图像转换为 PyTorch Tensor
        tensor_frame = self.transform(frame)
        tensor_frame = tensor_frame.unsqueeze(0)  # 添加 batch 维度

        # 使用模型进行推理
        with torch.no_grad():
            output = self.model(tensor_frame)['out'][0]  # 获取输出
            output_predictions = output.argmax(0)  # 获取每个像素的预测类别

        return output_predictions.cpu().numpy()  # 返回 NumPy 数组
    # threading 2
    def run(self):
        while rclpy.ok():
            # 获取锁：进入 with self.frame_lock: 时，程序尝试获取锁。如果 capture_frames 线程正在更新 self.latest_frame，那么此时 run 方法会等待，直到锁被释放。
            # 读取数据：成功获取锁后，frame 将被赋值为 self.latest_frame，这是最新捕获的图像。
            # 处理图像：在 if frame is not None: 条件下，如果成功读取到图像，程序就会调用 self.process_frame(frame) 进行处理。
            # 自动释放锁：当程序退出 with 块时，锁会被自动释放，允许其他线程（如 capture_frames）继续执行。
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                # 处理图像
                output = self.process_frame(frame)

                # 将输出数组转换为图像（可视化）
                output_image = (output * 255 / output.max()).astype('uint8')  # 归一化到 0-255
                output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_JET)  # 应用伪彩色

                # 将图像转换为 ROS 消息
                ros_image = self.br.cv2_to_imgmsg(output_image, encoding="bgr8")
                self.publisher.publish(ros_image)

                # 显示图像
                cv2.imshow("USB Camera", frame)
                cv2.imshow("Segmentation Output", output_image)

                if cv2.waitKey(30) & 0xFF == ord('q'):  # 调整等待时间
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