from PIL import Image
import numpy as np
# ---- Compute cluster probability + rate ----
def compute_rate(labels, K):
    prob = np.array([(labels == i).mean() for i in range(K)])
    prob = np.maximum(prob, 1e-12)   # avoid log(0)
    rate = -np.log2(prob)            # R = -log2(p)
    return rate

def rd_kmeans(pixels, K, lam=0.01, iters=10):
    N = len(pixels)
    
    idx = np.random.choice(N, K, replace=False)
    centroids = pixels[idx]
    labels = np.zeros(N, dtype=int)
    for _ in range(iters):
        
        rate = compute_rate(labels, K)
        
        D = np.sum((pixels[:, None] - centroids[None, :])**2, axis=2)
        
        RD_cost = D + lam * rate
        
        labels = np.argmin(RD_cost, axis=1)
        
        new_cent = []
        for k in range(K):
            pts = pixels[labels == k]
            if len(pts) == 0:
                new_cent.append(centroids[k])
            else:
                new_cent.append(pts.mean(axis=0))
        centroids = np.array(new_cent)
    return centroids, labels

def run(image_path, K, lam):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    pixels = arr.reshape(-1, 3).astype(float)
    cent, lbl = rd_kmeans(pixels, K, lam)
    out = cent[lbl].reshape(H, W, 3).astype(np.uint8)
    Image.fromarray(out).save("rd_output.jpg")
    print("Saved → rd_output.jpg")
K=int(input("Enter the no.of colors: "))
lam=float(input("Enter trade-off factor: "))
run("input.jpg", K, lam)