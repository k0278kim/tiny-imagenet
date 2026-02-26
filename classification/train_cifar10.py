import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # 추가
from tqdm import tqdm
import argparse
import os

# 사용자 정의 모듈 임포트
from resnet50_1_tinet import ResNet, Bottleneck
# from nets.early_stopping import EarlyStopping # 스크린샷 가이드(200회 완주)를 위해 사용 여부 선택

# 1. 인자 설정
parser = argparse.ArgumentParser(description='ResNet-50 CIFAR-10 Training (From Scratch)')
parser.add_argument('--cusin', type=int, default=1, help='custom convolution layer index')
args = parser.parse_args()

# 2. 하이퍼파라미터 (스크린샷 기준 업데이트)
BATCH_SIZE = 128
NUM_EPOCHS = 200
INITIAL_LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 3. 데이터 전처리 및 증강 (스크린샷 1번 항목 반영)
stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=stats[1])
])

# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 4. 모델 설정 (CIFAR-10 최적화)
# num_classes를 10으로 변경
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, custom_conv_layer_index=args.cusin).to(device)

# [중요] CIFAR-10용 입력 레이어 수정 
# 이미지 크기가 32x32이므로 첫 7x7 conv와 maxpool을 수정해야 정보 손실이 없습니다.
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
model.bn1 = nn.Identity() # maxpool 제거 (Identity로 대체)
model = model.to(device)

checkpoint_path = f'best_cifar10_cusin_{args.cusin}.pth'

if os.path.exists(checkpoint_path):
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    # map_location은 GPU/CPU 환경이 달라도 안전하게 로드하기 위해 사용합니다.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully. Resuming training...")
else:
    print("🆕 No checkpoint found. Starting from scratch.")

# 5. 손실 함수 및 옵티마이저 (스크린샷 2, 5번 항목)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, 
                      weight_decay=WEIGHT_DECAY, nesterov=True)

# 6. 스케줄러 (스크린샷 4번 Option B 반영)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# 7. 학습 루프
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # --- [TRAIN PHASE] ---
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.5f}"})

    # --- [VALIDATION PHASE] ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f"📊 Result: Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2f}%")

    # 스케줄러 업데이트
    scheduler.step()

    # 최고 정확도 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'best_cifar10_cusin_{args.cusin}.pth')
        print(f"🌟 Best Model Saved! (Acc: {best_acc:.2f}%)")

print(f"🏁 Final Best Accuracy: {best_acc:.2f}%")