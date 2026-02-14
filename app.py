import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PIL import Image

# ==========================================
# 1. 页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="Mesopic Vision Pro", 
    layout="wide", 
    page_icon="🌑"
)

# 强制深色风格 CSS & 卡片样式 & 优化后的滑动条
st.markdown("""
<style>
    /* 1. 整体背景与文字 */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* 2. 侧边栏背景 */
    [data-testid="stSidebar"] {
        background-color: #1A1C24; /*稍微加深一点，更有质感*/
        border-right: 1px solid #2D2F3B;
    }
    
    /* 3. 数据卡片样式 */
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-label {
        font-size: 12px;
        color: #9CA3AF;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 20px; /* 稍微放大数值 */
        font-weight: 700;
        color: #F3F4F6;
        font-family: 'Segoe UI', monospace;
    }
    .metric-sub {
        font-size: 11px;
        color: #6B7280;
        margin-top: 4px;
    }
    .color-preview {
        height: 60px;
        width: 100%;
        border-radius: 6px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* =========== 4. 滑动条终极美化 (和谐蓝调) =========== */
    
    /* 滑动条容器稍微增加间距 */
    div[data-testid="stSlider"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* (A) 滑块手柄 - 改为白色圆点带蓝色光晕，更显精致 */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #FFFFFF !important; /* 纯白手柄 */
        border: 2px solid #3B82F6 !important; /* 蓝色边框 */
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.4) !important; /* 蓝色呼吸光 */
        height: 18px !important; /* 稍微调大一点便于拖拽 */
        width: 18px !important;
    }

    /* (B) 已填充的轨道 - 科技蓝 */
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        background-color: #F3F4F6 !important;
        height: 6px !important; /* 轨道变细一点，显得更现代 */
        border-radius: 3px;
    }
    
    /* (C) 未填充的轨道背景 - 深灰 */
    div.stSlider > div[data-baseweb="slider"] > div > div {
         background-color: #4B5563 !important;
         height: 6px !important;
         border-radius: 3px;
    }

    /* (D) 数值文字 - 改为淡灰色，不要用彩色，避免喧宾夺主 */
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #E5E7EB !important; 
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心算法 (保持不变)
# ==========================================
class MesopicModel:
    def __init__(self, xp, yp, Lp, Ls, m):
        self.xp, self.yp = xp, yp
        self.Lp, self.Ls = Lp, Ls
        self.m = m

    def calculate(self):
        m, Lp, Ls, xp, yp = self.m, self.Lp, self.Ls, self.xp, self.yp
        if Lp >= 5.0:
            m = 1.0  # 强制进入纯明视觉
        elif Ls <= 0.005:
            m = 0.0  # 强制进入纯暗视觉
        safe_yp = max(yp, 0.0001)
        ratio_K = 683.0 / 1699.0
        
        num_L = m * Lp + (1 - m) * Ls * ratio_K
        den_L = m + (1 - m) * ratio_K
        if den_L == 0: den_L = 0.0001
        Lmes = num_L / den_L
        
        Lpa, Lsa = Lmes * m, Lmes * (1 - m)
        term_p, term_s = Lpa / safe_yp, Lsa / 0.3333
        denom = term_p + term_s
        if denom == 0: denom = 0.0001
        
        xm = (Lpa * xp / safe_yp + Lsa) / denom
        ym = (Lpa + Lsa) / denom
        return xm, ym, Lmes

    def xyY_to_rgb(self, x, y, Y):
        if y <= 1e-5: return '#000000'
        X = (x * Y) / y
        Z = ((1 - x - y) * Y) / y
        r =  3.2406*X - 1.5372*Y - 0.4986*Z
        g = -0.9689*X + 1.8758*Y + 0.0415*Z
        b =  0.0557*X - 0.2040*Y + 1.0570*Z
        rgb = [max(0, c)**(1/2.2) for c in (r, g, b)]
        m_val = max(rgb)
        if m_val > 1: rgb = [c/m_val for c in rgb]
        else: rgb = [min(1, c) for c in rgb]
        return '#{:02x}{:02x}{:02x}'.format(*(int(c*255) for c in rgb))

    def srgb_to_xyY_image(self, img_array):
        norm_img = img_array / 255.0
        mask = norm_img > 0.04045
        lin_img = np.where(mask, ((norm_img + 0.055) / 1.055) ** 2.4, norm_img / 12.92)
        R, G, B = lin_img[..., 0], lin_img[..., 1], lin_img[..., 2]
        X = 0.4124 * R + 0.3576 * G + 0.1805 * B
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Z = 0.0193 * R + 0.1192 * G + 0.9505 * B
        sum_XYZ = X + Y + Z
        sum_XYZ[sum_XYZ == 0] = 1e-6
        x = X / sum_XYZ
        y = Y / sum_XYZ
        return x, y, Y

    def calculate_image(self, m, Lp_img, Ls_img, xp_img, yp_img):
        safe_yp = np.maximum(yp_img, 0.0001)
        ratio_K = 683.0 / 1699.0
        # 创建一个与 Lp_img 维度相同的 m 值矩阵
        m_map = np.full_like(Lp_img, m)
        # 根据每个像素的真实亮度进行覆盖
        m_map = np.where(Lp_img >= 5.0, 1.0, m_map)
        m_map = np.where(Ls_img <= 0.005, 0.0, m_map)
        num_L = m_map * Lp_img + (1 - m_map) * Ls_img * ratio_K
        den_L = m_map + (1 - m_map) * ratio_K
        Lmes = num_L / den_L
        
        Lpa, Lsa = Lmes * m_map, Lmes * (1 - m_map)
        term_p = Lpa / safe_yp
        term_s = Lsa / 0.3333
        denom = term_p + term_s
        denom = np.where(denom == 0, 1e-6, denom)
        
        xm = (Lpa * xp_img / safe_yp + Lsa) / denom
        ym = (Lpa + Lsa) / denom
        return xm, ym, Lmes

    def xyY_to_srgb_image(self, x, y, Y):
        y = np.maximum(y, 1e-6)
        X = (x * Y) / y
        Z = ((1 - x - y) * Y) / y
        R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
        rgb = np.dstack((R, G, B))
        rgb = np.clip(rgb, 0, 1)
        mask = rgb > 0.0031308
        srgb = np.where(mask, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055, 12.92 * rgb)
        return np.clip(srgb * 255, 0, 255).astype(np.uint8)

# ==========================================
# 3. 缓存背景 (Matplotlib 深色模式适配)
# ==========================================
@st.cache_data
def get_cie_background():
    # 使用 Dark Background 样式
    plt.style.use('dark_background')
    
    # 完整的光谱轨迹坐标
    spectral_locus = np.array([
        [0.1741, 0.0050], [0.1740, 0.0050], [0.1738, 0.0049], [0.1736, 0.0049], [0.1733, 0.0048],
        [0.1730, 0.0048], [0.1726, 0.0048], [0.1721, 0.0048], [0.1714, 0.0051], [0.1703, 0.0058],
        [0.1689, 0.0069], [0.1669, 0.0086], [0.1644, 0.0109], [0.1611, 0.0138], [0.1566, 0.0177],
        [0.1510, 0.0227], [0.1440, 0.0297], [0.1355, 0.0399], [0.1241, 0.0578], [0.1096, 0.0868],
        [0.0913, 0.1327], [0.0687, 0.2007], [0.0454, 0.2950], [0.0235, 0.4127], [0.0082, 0.5384],
        [0.0039, 0.6548], [0.0139, 0.7502], [0.0389, 0.8120], [0.0743, 0.8338], [0.1142, 0.8262],
        [0.1547, 0.8059], [0.1929, 0.7816], [0.2296, 0.7543], [0.2658, 0.7243], [0.3016, 0.6923],
        [0.3373, 0.6589], [0.3731, 0.6245], [0.4087, 0.5896], [0.4441, 0.5547], [0.4788, 0.5202],
        [0.5125, 0.4866], [0.5448, 0.4544], [0.5752, 0.4242], [0.6029, 0.3965], [0.6270, 0.3725],
        [0.6482, 0.3514], [0.6658, 0.3340], [0.6801, 0.3197], [0.6915, 0.3083], [0.7006, 0.2993],
        [0.7079, 0.2920], [0.7140, 0.2859], [0.7190, 0.2809], [0.7230, 0.2770], [0.7260, 0.2740],
        [0.7283, 0.2717], [0.7300, 0.2700], [0.7311, 0.2689], [0.7320, 0.2680], [0.7327, 0.2673],
        [0.7334, 0.2666], [0.7340, 0.2660], [0.7344, 0.2656], [0.7346, 0.2654], [0.7347, 0.2653]
    ])
    # 闭合曲线
    spectral_locus = np.vstack([spectral_locus, spectral_locus[0]])
    
    resolution = 400
    x = np.linspace(0, 0.8, resolution)
    y = np.linspace(0, 0.9, resolution)
    xx, yy = np.meshgrid(x, y)
    
    path = Path(spectral_locus)
    mask = path.contains_points(np.vstack((xx.flatten(), yy.flatten())).T).reshape(resolution, resolution)
    
    X, Y, Z = xx, yy, 1 - xx - yy
    R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    RGB = np.clip(np.dstack((R, G, B)), 0, 1) ** (1/2.2)
    RGBA = np.zeros((resolution, resolution, 4))
    RGBA[..., 0:3] = RGB
    RGBA[..., 3] = mask.astype(float)
    return RGBA, spectral_locus

# ==========================================
# 4. 侧边栏布局 (Left Panel)
# ==========================================
with st.sidebar:
    st.markdown("### Mesopic Vision")
    st.markdown("<div style='color:#666; margin-bottom:20px'>Control Panel</div>", unsafe_allow_html=True)
    
    # 1. 参数输入 (修复了 value > max_value 的错误)
    c1, c2 = st.columns(2)
    with c1:
        xp = st.number_input("x", min_value=0.0, max_value=1.0, value=0.45, step=0.01, format="%.3f")
        Lp = st.number_input("Lp", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
    with c2:
        yp = st.number_input("y", min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.3f")
        Ls = st.number_input("Ls", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        
    st.write("")
    m = st.slider("Adaptation (m)", 0.0, 1.0, 1.0, 0.01)
    
    st.divider()
    if Lp >= 5.0:
        st.info("💡 Lp ≥ 5.0, 强制为明视觉 (m=1.0)")
    elif Ls <= 0.005:
        st.info("🌑 Ls ≤ 0.005, 强制为暗视觉 (m=0.0)")
    # 2. 实时计算
    model = MesopicModel(xp, yp, Lp, Ls, m)
    xm, ym, Lmes = model.calculate()
    hex_p = model.xyY_to_rgb(xp, yp, Lp)
    hex_m = model.xyY_to_rgb(xm, ym, Lmes)
    
    # 3. CIE 图表 (放在左侧)
    cie_img, locus = get_cie_background()
    
    # 创建深色背景的图表
    fig, ax = plt.subplots(figsize=(4, 4))
    # 设置图表背景透明，以适应网页深色背景
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # 绘图
    ax.imshow(cie_img, origin='lower', extent=[0, 0.8, 0, 0.9], interpolation='bicubic', zorder=0)
    ax.plot(locus[:, 0], locus[:, 1], 'w-', linewidth=0.8, zorder=1) # 白色轮廓线
    ax.plot([xp, xm], [yp, ym], 'w--', lw=1, alpha=0.5, zorder=5) # 白色虚线
    
    ax.plot(xp, yp, 'o', ms=8, mfc='#3B82F6', mec='white', mew=1.5, label='P', zorder=10) # 蓝色 P
    ax.plot(xm, ym, '^', ms=8, mfc='#F59E0B', mec='white', mew=1.5, label='M', zorder=10) # 橙色 M
    
    ax.set_title("CIE Chromaticity Diagram", color="white", fontsize=10, pad=10)
    ax.tick_params(colors='white', labelsize=7)
    # 隐藏边框 spine
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
        
    ax.legend(loc='upper right', frameon=False, fontsize=8, labelcolor='white')
    st.pyplot(fig)

# ==========================================
# 5. 主区域布局 (Right Panel)
# ==========================================

# 顶部：上传控件
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'bmp'], label_visibility="collapsed")

if uploaded_file is None:
    # 占位状态
    st.markdown("""
    <div style="
        border: 2px dashed #374151; 
        border-radius: 10px; 
        height: 600px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        background-color: #111827;
        color: #6B7280;
    ">
        <h3>📂 Drag and drop or click to upload an image</h3>
    </div>
    """, unsafe_allow_html=True)
else:
    # 处理图片
    image = Image.open(uploaded_file).convert('RGB')
    img_arr = np.array(image)
    
    with st.spinner('Calculating...'):
        xp_img, yp_img, Yp_img = model.srgb_to_xyY_image(img_arr)
        
        avg_Y = np.mean(Yp_img)
        if avg_Y == 0: avg_Y = 0.001
        scaling_factor = Lp / avg_Y
        Lp_map = Yp_img * scaling_factor
        sp_ratio = Ls / Lp
        Ls_map = Lp_map * sp_ratio
        
        xm_img, ym_img, Lmes_img = model.calculate_image(m, Lp_map, Ls_map, xp_img, yp_img)
        
        Y_out_display = Lmes_img / scaling_factor
        processed_arr = model.xyY_to_srgb_image(xm_img, ym_img, Y_out_display)
        result_image = Image.fromarray(processed_arr)

    # === 结果展示区域 (修改：图片在上，色块在下) ===
    
    # 1. 图片对比 (左右对比，最大化利用空间)
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.caption("Original Scene")
        st.image(image, use_container_width=True)
    with c_img2:
        st.caption(f"Simulation (m={m:.2f})")
        st.image(result_image, use_container_width=True)
        
    # 2. 数据卡片 (模仿截图右侧的数据卡片，放在图片下方)
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # 使用 HTML 渲染自定义卡片
        st.markdown(f"""
        <div class="metric-card">
            <div class="color-preview" style="background-color: {hex_p}"></div>
            <div class="metric-label" style="color:#3B82F6">Original (Photopic)</div>
            <div class="metric-value">L = {Lp:.2f}</div>
            <div class="metric-sub">x:{xp:.3f} y:{yp:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="color-preview" style="background-color: {hex_m}"></div>
            <div class="metric-label" style="color:#F59E0B">Predicted (Mesopic)</div>
            <div class="metric-value">L = {Lmes:.2f}</div>
            <div class="metric-sub">x:{xm:.3f} y:{ym:.3f}</div>
        </div>

        """, unsafe_allow_html=True)

