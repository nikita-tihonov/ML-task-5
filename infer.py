from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

class FaceRecognition:

    def __init__(self):
        self.detector = MTCNN(
            keep_all=True,
            device="cuda"
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_face(self, image):
        boxes, probs = self.detector.detect(image, landmarks=False)
        if boxes is not None:
            box = boxes[0]
            image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            return image

        return None


    def processing_photo(self, image):
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor


    def get_embedding(self, image):
        image = self.extract_face(image)
        image = self.processing_photo(image)
        with torch.no_grad():
            embedding = self.model(image)

        return embedding


    @staticmethod
    def get_cosine_similarity(emb_1, emb_2):
        emb_1 = emb_1.reshape(1, -1)
        emb_2 = emb_2.reshape(1, -1)
        return cosine_similarity(emb_1, emb_2)[0][0]

if __name__ == '__main__':

    face_recognition = FaceRecognition()

    sample = cv2.imread('sample.jpeg')
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample_emb = face_recognition.get_embedding(sample)

    my_photos = []
    for filename in os.listdir("my_photos"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join("my_photos", filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            my_photos.append(img)

    other_photos = []
    for filename in os.listdir("other_photos"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join("other_photos", filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            other_photos.append(img)

    my_embs = []
    for img in my_photos:
        emb = face_recognition.get_embedding(img)
        my_embs.append(emb)

    other_embs = []
    for img in other_photos:
        emb = face_recognition.get_embedding(img)
        other_embs.append(emb)

    fig, axes = plt.subplots(4, 5, figsize=(10, 10))

    for ax in axes.flat:
        ax.axis('off')

    axes[0][0].imshow(sample)
    axes[0][0].set_title("Образец")


    for i, img in enumerate(my_photos):
        ax = axes[1][i]
        ax.imshow(my_photos[i])
        similarity = face_recognition.get_cosine_similarity(sample_emb, my_embs[i])
        res = "True" if similarity > 0.6 else "False"
        ax.set_title(f"{res}, Similarity: {similarity:.2f}")

    for i, img in enumerate(other_photos):
        ax = axes[i//5 + 2][i%5]
        ax.imshow(other_photos[i])
        similarity = face_recognition.get_cosine_similarity(sample_emb, other_embs[i])
        res = "True" if similarity > 0.6 else "False"
        ax.set_title(f"{res}, Similarity: {similarity:.2f}")

    plt.tight_layout()

    plt.savefig('result.png')

    torch.save(InceptionResnetV1(pretrained="vggface2").state_dict(), 'lfw_face_embedding_model.pth')


