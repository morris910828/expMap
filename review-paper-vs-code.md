## 不一致（Discrepancies）

### D1. `_distance` 未套用 sigmoid 啟動函數

- 在 [scene/dual_gaussian_model.py:41](scene/dual_gaussian_model.py#L41) 定義 `self.distance_activation = torch.sigmoid`，但實際使用處：
  - [scene/dual_gaussian_model.py:122-123](scene/dual_gaussian_model.py#L122-L123) 的 `get_distance` 直接回傳 `self._distance`。
  - [scene/dual_gaussian_model.py:172](scene/dual_gaussian_model.py#L172) 計算位置時 `offset = self._distance * n`。
- `distance_activation` 在整個 repo 從未被呼叫（僅在 `__setup_functions` 內定義）。
- 影響：η 沒有非負/有界限制，可正可負。論文 Eq 3-3、3-6 沒有顯式說明 η 的啟動函數，但 sigmoid 既被預設定義，疑似遺漏呼叫端。

### D2. Distance threshold `d₀` 數值差 10×

- 論文 4.2「實驗設定」明確寫 **d₀ = 0.01**。
- 程式碼 [train.py:330](train.py#L330) 寫死：
  ```python
  dis_loss = torch.mean(torch.clamp(gaussians.get_distance - 0.001, min=0) ** 2)
  ```
- 差距：code 的 `0.001` 比論文嚴格 10 倍；且為 hard-code，未在 [arguments/__init__.py](arguments/__init__.py) 暴露為 `OptimizationParams` 欄位。

### D3. Regularization Loss 採三角形外接圓半徑而非頂點半徑

- 論文 Eq 3-5：`Lr = Σ max(max(s_g) − R_g, 0)`，文中 R_g 指 mesh 上對應頂點的局部半徑。
- 程式碼 [utils/loss_utils.py](utils/loss_utils.py) 的 `mesh_restrict_loss` 以 **三角形 circumradius**（外接圓半徑，由 `vert1, vert2, vert3` 計算）作為上界，呼叫於 [train.py:332-333](train.py#L332-L333)。
- 而 [utils/general_utils.py](utils/general_utils.py) 的 `compute_vertex_radii_approx` 確實計算了「頂點到 1-ring 鄰居的最大距離」並存入 [scene/dual_gaussian_model.py:260](scene/dual_gaussian_model.py#L260) `self.vertex_radius`，但**從未被任何 loss 引用**（屬於 dead state）。
- 影響：circumradius 通常大於 vertex 1-ring 半徑，會讓 reg loss 較鬆，與論文字面定義不一致。

---

## 不一致造成的結果差異分析

### D1 影響 — η 可正可負且無上界

**理論差異**：
- 論文設計（有 sigmoid）：η ∈ (0, 1)，**強制為正**，appearance Gaussian 只能位於 mesh 表面**外側**沿法向 n 的方向。
- 目前實作（無 sigmoid）：`_distance` 為無界連續值 (−∞, +∞)，η **可正可負**。

**可能造成的後果**：
1. **Appearance Gaussian 漂進 mesh 內部**：當 `_distance` 為負時，Gaussian 中心會落到 mesh 內側。
2. **Distance loss 失效於負側**：[train.py:330](train.py#L330) 是 `clamp(d − 0.001, 0)²`，**只懲罰 d > 0.001**；負值完全沒有懲罰。Gaussian 可以無限制地往內偏移。
3. **變形後出現破面**：[deformation.py](deformation.py) 把 η 當作 rigid 法向偏移傳播；若原本就有 Gaussian 跑進內部，mesh 形變後會出現「殼內漏點」造成的 artifact，與論文 5.4 節宣稱「mesh 表面緊貼」的訴求衝突。
4. **與論文消融研究的數值對不上**：論文 5.3 表 5-4 顯示加上 appearance Gaussian 後 PSNR 普遍提升 ~1 dB，這份數據預期是在 sigmoid 約束下取得；目前實作可能因 Gaussian 散得更廣、更自由，導致**訓練 PSNR 看起來甚至略高**，但 mesh-aligned 的物理一致性下降。

### D2 影響 — d₀ 嚴格 10×

**理論差異**：
- 論文 d₀ = 0.01：appearance Gaussian 允許在 mesh 表面外 1 cm（假設世界座標單位為 m）以內自由移動而不受懲罰。
- 實作 d₀ = 0.001：只允許 1 mm 緩衝，超出即受平方懲罰。

**可能造成的後果**：
1. **損失大量「表面外觀層」表現力**：頭髮、絨毛、半透明邊緣、薄殼物件（論文用的 Bunny / Lucy / Dragon 都有這類細節）需要 Gaussian 在表面附近、但**有一定外擴**才能渲染好。1 mm 太緊，會壓死這個自由度。
2. **PSNR 較論文略低**：對照論文 5.1 表 5-1（Ours），訓練集 PSNR 通常 33–35；用 d₀=0.001 應該會掉 0.3–1 dB（取決於物件，毛邊類更明顯）。Burger（PSNR 29）這種有複雜表面紋理的更敏感。
3. **與 D1 互相加劇**：D1 讓 d 可為負（無懲罰），D2 又把正側壓得更緊。優化會傾向把 `_distance` 推到 0 甚至負側，**等於完全靠近 mesh 表面**，appearance 退化成幾乎只是 geometry Gaussian 的「貼皮」。論文設計 appearance Gaussian 的初衷（補足 mesh 無法表達的外觀細節）會被部分抵銷。

### D3 影響 — circumradius vs vertex 1-ring radius

**理論差異**：

| 量 | 公式 | 等邊三角形（邊長 a） | 細長三角形 |
|---|---|---|---|
| Circumradius | abc / (4·Area) | a/√3 ≈ 0.577a | 可膨脹至 ≫ a |
| Vertex 1-ring radius | max ‖v − v_neighbor‖ | a | a |

- 等邊三角形：circumradius **較小**（更嚴格上界）
- 鈍角／細長三角形：circumradius **可暴增**（幾乎無約束）

**可能造成的後果**：
1. **Mesh 品質敏感度提高**：論文用的 Objaverse 物件（Chair / Burger / Toy）三角網品質不一致；遇到細長三角形（常見於 UV 接縫、薄板邊緣），Gaussian 尺寸完全不被限制，會出現超大「漏球」。
2. **均勻網格上反而過嚴**：等邊網格區域，circumradius 比 1-ring 半徑小 ~40%，Gaussian 被壓得太小，**重建細節能力下降**。可能對應論文 5.1 中 Chair（PSNR 30.50）這類有平面區的物件特別影響。
3. **`vertex_radius` 是 dead state**：[scene/dual_gaussian_model.py:260](scene/dual_gaussian_model.py#L260) 計算了卻沒用，意味著模型 `.ply` 檔可能含有未使用的中間量，並無實際 loss 訊號。
4. **變形階段一致性**：Gaussian 在變形時用 [deformation.py:80-81](deformation.py#L80-L81) 的重心內插 scale，**訓練時的 size 約束**直接影響變形後 Gaussian 是否會穿出新 mesh 表面。circumradius 約束在大三角區域過鬆，會讓變形後的「Gaussian 雲」邊緣鬆散感更明顯。

### 三者交互作用

對照論文 5.1（訓練集）與 5.2（測試集）的 PSNR/SSIM 數值若無法復現，最可能的解釋順序：

1. **D2 直接造成 ~0.3–1 dB PSNR 下降**（最易測）
2. **D1 造成 mesh 形變後外觀出現「內陷 artifact」**（看 deformation 輸出最明顯）
3. **D3 造成不同物件 PSNR 落差不一致**（Chair 偏低、有 UV 縫的物件偏高且雜訊多）

**驗證方式**：若想實證，最有效率的做法是只改 D2（一行字面值），重訓 Burger / Lucy 各一個並比 PSNR；若改完 D2 仍復現不出論文數值，再進一步處理 D1 與 D3。


