import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
from function import adaptive_instance_normalization as adain
from net import decoder, vgg, Net

def load_image(path, device, image_size=512):
    """Завантажує зображення, змінює розмір і перетворює у тензор."""
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def denormalize(tensor, mean, std):
    """Денормалізація тензора за середніми і std."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def stylize_image(content_path, style_path, output_path, alpha=1.0, image_size=512):
    """
    Виконує стилізацію content-зображення за стилем style-зображення
    з інтенсивністю alpha.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується пристрій: {device}")

    # Завантажуємо і готуємо зображення
    content = load_image(content_path, device, image_size)
    style = load_image(style_path, device, image_size)

    # Нормалізація для VGG (за середнім та std ImageNet)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3,1,1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3,1,1)

    vgg_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    content_norm = vgg_normalize(content.squeeze(0)).unsqueeze(0)
    style_norm = vgg_normalize(style.squeeze(0)).unsqueeze(0)

    # Ініціалізація моделей
    print("Завантаження моделей...")
    dec = decoder.to(device)
    dec_state = torch.load('models/decoder_iter_160000.pth.tar', map_location=device)
    dec.load_state_dict(dec_state)
    dec.eval()

    vgg_model = vgg
    vgg_state = torch.load('models/vgg_normalised.pth', map_location=device)
    vgg_model.load_state_dict(vgg_state)
    vgg_model.eval()
    vgg_model.to(device)

    model = Net(vgg_model, dec)
    model.eval().to(device)

    # Стилізація
    print("Стилізація...")
    with torch.no_grad():
        _, _, output = model(content_norm, style_norm, alpha=alpha)

    # Денормалізація тензора (повертаємо кольори у нормальний діапазон)
    output_denorm = output.squeeze(0).cpu().clone()
    output_denorm = denormalize(output_denorm, imagenet_mean.cpu(), imagenet_std.cpu())
    output_denorm = output_denorm.clamp(0, 1)

    # Конвертація в PIL Image
    img = transforms.ToPILImage()(output_denorm)

    # Опціонально: зменшити контраст трохи (щоб уникнути "кислотних" кольорів)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.9)  # 0.9 - трохи менший контраст (регулюй за потребою)

    # Збереження результату
    img.save(output_path)
    print(f"Результат збережено у {output_path}")


if __name__ == "__main__":
    # Тест запуску скрипта з параметрами (замініть шляхи на свої)
    stylize_image(
        content_path='path/to/content.jpg',
        style_path='path/to/style.jpg',
        output_path='path/to/output.jpg',
        alpha=0.8,
        image_size=512
    )
