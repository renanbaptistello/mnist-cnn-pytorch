from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results" / "model.pth"

WINDOW_WIDTH = 760
WINDOW_HEIGHT = 520
CANVAS_SIZE = 280
CANVAS_POS = (40, 120)
BRUSH_RADIUS = 7
BG_COLOR = (245, 247, 250)
TEXT_COLOR = (25, 30, 40)
CANVAS_BORDER = (60, 65, 75)
DRAW_COLOR = (255, 255, 255)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em {MODEL_PATH}")

    model = Net()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def surface_to_tensor(surface):
    raw = pygame.surfarray.array3d(surface)
    grayscale = raw[:, :, 0].T
    image = Image.fromarray(grayscale, mode="L")

    bbox = image.getbbox()
    if bbox is None:
        return None

    image = image.crop(bbox)
    image = ImageOps.contain(image, (20, 20), method=Image.Resampling.LANCZOS)

    final_image = Image.new("L", (28, 28), 0)
    offset = ((28 - image.width) // 2, (28 - image.height) // 2)
    final_image.paste(image, offset)

    array = np.asarray(final_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081
    return tensor


def predict_digit(model, canvas_surface):
    tensor = surface_to_tensor(canvas_surface)
    if tensor is None:
        return None, None

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.exp(output)[0]
        digit = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[digit].item() * 100.0)
    return digit, confidence


def draw_text(screen, font, text, color, pos):
    screen.blit(font.render(text, True, color), pos)


def main():
    pygame.init()
    pygame.display.set_caption("Reconhecimento de Digitos MNIST")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("arial", 30, bold=True)
    body_font = pygame.font.SysFont("arial", 22)
    small_font = pygame.font.SysFont("arial", 18)

    canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas_surface.fill((0, 0, 0))

    model = load_model()
    prediction = None
    confidence = None
    drawing = False

    predict_button = pygame.Rect(390, 180, 290, 52)
    clear_button = pygame.Rect(390, 250, 290, 52)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    canvas_surface.fill((0, 0, 0))
                    prediction = None
                    confidence = None
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    prediction, confidence = predict_digit(model, canvas_surface)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(CANVAS_POS, (CANVAS_SIZE, CANVAS_SIZE)).collidepoint(event.pos):
                    drawing = True
                elif predict_button.collidepoint(event.pos):
                    prediction, confidence = predict_digit(model, canvas_surface)
                elif clear_button.collidepoint(event.pos):
                    canvas_surface.fill((0, 0, 0))
                    prediction = None
                    confidence = None

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False

            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                local_x = x - CANVAS_POS[0]
                local_y = y - CANVAS_POS[1]
                if 0 <= local_x < CANVAS_SIZE and 0 <= local_y < CANVAS_SIZE:
                    pygame.draw.circle(canvas_surface, DRAW_COLOR, (local_x, local_y), BRUSH_RADIUS)

        if drawing and pygame.Rect(CANVAS_POS, (CANVAS_SIZE, CANVAS_SIZE)).collidepoint(mouse_pos):
            local_x = mouse_pos[0] - CANVAS_POS[0]
            local_y = mouse_pos[1] - CANVAS_POS[1]
            pygame.draw.circle(canvas_surface, DRAW_COLOR, (local_x, local_y), BRUSH_RADIUS)

        screen.fill(BG_COLOR)

        draw_text(screen, title_font, "Desenhe um numero e clique em Prever", TEXT_COLOR, (40, 35))
        draw_text(screen, small_font, "Enter/Espaco: prever   C: limpar", (90, 95, 110), (40, 75))

        canvas_rect = pygame.Rect(CANVAS_POS, (CANVAS_SIZE, CANVAS_SIZE))
        pygame.draw.rect(screen, CANVAS_BORDER, canvas_rect.inflate(6, 6), border_radius=8)
        screen.blit(canvas_surface, CANVAS_POS)

        pygame.draw.rect(screen, (50, 125, 230), predict_button, border_radius=8)
        pygame.draw.rect(screen, (210, 70, 70), clear_button, border_radius=8)
        draw_text(screen, body_font, "Prever", (255, 255, 255), (predict_button.x + 108, predict_button.y + 11))
        draw_text(screen, body_font, "Limpar", (255, 255, 255), (clear_button.x + 105, clear_button.y + 11))

        result_box = pygame.Rect(390, 335, 290, 105)
        pygame.draw.rect(screen, (225, 230, 238), result_box, border_radius=8)

        if prediction is None:
            draw_text(screen, body_font, "Predicao: --", TEXT_COLOR, (410, 360))
            draw_text(screen, small_font, "Desenhe um digito branco em fundo preto.", (90, 95, 110), (410, 398))
        else:
            draw_text(screen, title_font, f"Predicao: {prediction}", TEXT_COLOR, (410, 352))
            draw_text(screen, body_font, f"Confianca: {confidence:.2f}%", (55, 90, 155), (410, 394))

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
