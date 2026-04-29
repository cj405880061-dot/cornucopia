import os
# --- 强力网络装甲：强制清理系统环境变量中的错误代理设置 ---
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
# --------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import openai 
import base64
import httpx # 引入 httpx 用于构建无代理直连通道

# ==========================================
# 0. 页面全局设置
# ==========================================
st.set_page_config(layout="wide", page_title="油气田智能 Copilot 平台")

# ==========================================
# 1. 平台级状态初始化 (双系统记忆隔离)
# ==========================================
if "messages_rod" not in st.session_state:
    st.session_state.messages_rod = []
if "messages_card" not in st.session_state:
    st.session_state.messages_card = []
if "is_calculated" not in st.session_state:
    st.session_state.is_calculated = False

# ==========================================
# 2. 全局侧边栏：导航、密钥与模型调度中心
# ==========================================
with st.sidebar:
    st.title("🛢️ 智能采油平台")
    
    # 预埋专属 API 密钥
    api_key = st.text_input(
        "🔑 硅基流动 API Key", 
        type="password", 
        value="sk-nsfwofakxxyjonmemqmmadqspbviemrscabhqeqgpcp",
        help="已预埋专属密钥，可直接使用。也可手动修改。"
    )
    
    st.subheader("🧠 AI 模型调度中心")
    
    # --- 文本大模型 ---
    text_model = st.selectbox(
        "📝 文本大模型 (用于管柱与力学诊断)", 
        [
            "Qwen/Qwen2.5-72B-Instruct",   # 默认首选：阿里开源王者，极速且严谨
            "deepseek-ai/DeepSeek-V3",     # 旗舰备选：综合能力极强
            "deepseek-ai/DeepSeek-R1",     # 深度推演：遇到极复杂工况时选用（较慢）
            "Qwen/Qwen2.5-7B-Instruct"     # 免费保底：轻量级，100%不收费
        ],
        index=0, 
        help="推荐默认使用 Qwen2.5-72B，兼顾速度与工程逻辑精度。"
    )
    
    # --- 视觉大模型 ---
    vision_model = st.selectbox(
        "👁️ 视觉多模态模型 (用于功图诊断)", 
        [
            "Qwen/Qwen3-VL-8B-Instruct",      # 保底首选：Qwen 第3代最新 8B 小模型，防风控
            "zai-org/GLM-4.6V",               # 跨平台备选：智谱视觉大模型
            "Pro/Qwen/Qwen2.5-VL-7B-Instruct" # 专线备选：小参数 VIP 通道
        ],
        index=0,
        help="如遇报错说明账号权限不足，请尝试切换至下一个模型。"
    )
    
    st.divider()
    
    st.subheader("🧭 功能导航")
    app_mode = st.radio(
        "选择进入的工作空间：", 
        ["⚙️ 抽油杆多级受力诊断", "📈 智能功图图像识别"], 
        label_visibility="collapsed"
    )

# ==========================================
# 模块 A：抽油杆多级受力诊断系统
# ==========================================
if app_mode == "⚙️ 抽油杆多级受力诊断":
    st.title("⚙️ 抽油杆多级管柱综合诊断系统")
    
    # --- A1. 侧边栏参数配置 ---
    with st.sidebar:
        st.markdown("### 📋 杆柱参数配置")
        tubing_size = st.selectbox("油管规格 (外径/内径 mm)", ["73.0 / 62.0 (2 7/8\")", "88.9 / 76.0 (3 1/2\")", "60.3 / 50.3 (2 3/8\")"])
        tubing_id = float(tubing_size.split('/')[1].split('(')[0].strip()) 
        
        col_t1, col_t2 = st.columns(2)
        with col_t1: use_anchor = st.checkbox("🔧 设油管锚", value=True)
        with col_t2: use_echo = st.checkbox("📡 设回音标")
            
        fluid_density = st.number_input("混合液密度 (kg/m³)", value=980, step=10)
        centralizer_density = st.slider("扶正器密度 (个/100m)", 0, 10, 2)

        col_p1, col_p2 = st.columns(2)
        with col_p1: stroke_length = st.number_input("冲程 (m)", 1.0, 8.0, 3.0, 0.1)
        with col_p2: stroke_rate = st.number_input("冲次 (次/min)", 1, 15, 6, 1)
        pump_diameter = st.selectbox("泵径 (mm)", [32, 38, 44, 57, 70], index=2)
        
        st.markdown("### 🪜 多级配杆设计 (由下至上)")
        default_rod_data = pd.DataFrame({
            "段号": ["底部段 (接泵)", "中部段", "顶部段 (接悬点)"],
            "外径(mm)": [19.0, 22.0, 25.0],
            "长度(m)": [500.0, 500.0, 500.0],
            "钢级": ["D", "D", "H"]
        })
        rod_df = st.data_editor(default_rod_data, num_rows="dynamic", use_container_width=True, hide_index=True)
        
        btn_calc = st.button("🚀 运行受力校核", type="primary", use_container_width=True)
        if btn_calc: st.session_state.is_calculated = True

    # --- A2. 左侧主计算与 3D 渲染 ---
    col_main, col_ai = st.columns([7, 3], gap="large")
    with col_main:
        result_df = pd.DataFrame()
        total_depth = rod_df["长度(m)"].sum() if not rod_df.empty else 1500
        min_safety = 99
        current_context = ""
        fig = go.Figure()

        if st.session_state.is_calculated and not rod_df.empty:
            g, rho_steel = 9.81, 7850
            A_p = 3.14159 * (((pump_diameter / 2) / 1000) ** 2)
            omega = 2 * 3.14159 * stroke_rate / 60
            a_max = (stroke_length / 2) * (omega ** 2)
            W_f = A_p * total_depth * fluid_density * g * (1.0 if use_anchor else 0.85)
            
            cumulative_weight, cumulative_dynamic = 0, 0
            section_results = []
            rod_sections = rod_df.iloc[::-1].to_dict('records')
            
            for idx, row in enumerate(rod_sections):
                d_mm, L_m, grade = max(float(row["外径(mm)"]), 1), float(row["长度(m)"]), row["钢级"]
                A_r = 3.14159 * ((d_mm / 2) / 1000) ** 2
                W_section = (A_r * L_m * rho_steel * g) + ((L_m / 100) * centralizer_density * 2 * g)
                hydraulic_drag = (1 / max(tubing_id - d_mm, 1)) * 50 * L_m 
                F_d_section = W_section * (a_max / g)
                
                cumulative_weight += W_section
                cumulative_dynamic += F_d_section
                
                max_load_here = cumulative_weight + cumulative_dynamic + W_f + hydraulic_drag
                if use_echo and idx == len(rod_sections) - 1: max_load_here += 5 * g * (1 + a_max/g)
                stress_here = (max_load_here / A_r) / 1000000 
                
                allowable = {"C": 413, "D": 586, "K": 413, "H": 792}.get(str(grade)[0].upper(), 586) 
                safety_here = allowable / stress_here if stress_here > 0 else 99
                
                section_results.append({
                    "段落位置": f"第{idx+1}段(下至上)", "外径(mm)": d_mm, "长度(m)": L_m,
                    "顶载(kN)": round(max_load_here / 1000, 2), "应力(MPa)": round(stress_here, 2),
                    "安全系数": round(safety_here, 2)
                })

            result_df = pd.DataFrame(section_results[::-1])
            min_safety = result_df["安全系数"].min()

            current_z = 0
            for i, row in result_df.iterrows():
                L_m, d_mm, stress = row["长度(m)"], row["外径(mm)"], row["应力(MPa)"]
                z_raw = np.linspace(current_z, current_z + L_m, 20)
                theta = np.linspace(0, 2*np.pi, 20)
                theta_grid, z_grid = np.meshgrid(theta, z_raw)
                r_render = ((d_mm / 2) / 1000) * 8 
                x_grid, y_grid = r_render * np.cos(theta_grid), r_render * np.sin(theta_grid)
                color_grid = np.full(z_grid.shape, stress)
                fig.add_trace(go.Surface(
                    x=x_grid, y=y_grid, z=z_grid, 
                    surfacecolor=color_grid, colorscale='Jet', 
                    cmin=0, cmax=800, showscale=False
                ))
                current_z += L_m

            fig.update_layout(
                height=500, margin=dict(l=0, r=0, b=0, t=0), 
                scene=dict(zaxis_autorange="reversed", aspectmode='manual', aspectratio=dict(x=1, y=1, z=3))
            )
            
            st.subheader("📊 受力校核结果")
            st.dataframe(result_df, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)
            
            context_table = result_df.to_markdown(index=False)
            current_context = f"""
            你是一名极其严谨的油气田采油工程力学专家。
            【核心法则】
            你必须、只能参考下方提供的【实测计算数据表】，严禁自行推算或捏造任何安全系数或应力数值！

            【实测计算数据表】
            {context_table}
            
            【系统环境】总井深: {total_depth}m, 冲程: {stroke_length}m, 冲次: {stroke_rate}次/min。

            【回答强制结构】
            面对用户的提问，请严格按以下三步回答：
            1. [数据复核]：直接从表格中找出全井安全系数最低的段落，并报出它精确的安全系数值（保留两位小数）。
            2. [机理分析]：结合冲程/冲次和受力特点，分析为何该段最容易发生断脱或偏磨。
            3. [工程建议]：给出具有实操性的参数优化建议（如升级钢级、调整管径组合等）。
            """
        else:
            st.info("👈 请在左侧配置参数，并点击【运行受力校核】")

    # --- A3. 右侧理科 Copilot ---
    with col_ai:
        st.subheader(f"💬 结构优化 Copilot")
        st.caption(f"当前驱动引擎：{text_model.split('/')[-1]}")
        if st.button("🗑️ 清空力学对话", key="clear_rod"): 
            st.session_state.messages_rod = []
            st.rerun()
        
        chat_container = st.container(height=600)
        with chat_container:
            for msg in st.session_state.messages_rod:
                st.chat_message(msg["role"]).markdown(msg["content"])
                    
        if prompt := st.chat_input("向 AI 提问力学优化方案..."):
            if not api_key: 
                st.error("请确认 API Key 是否填写！")
            elif not st.session_state.is_calculated: 
                st.warning("请先运行受力校核！")
            else:
                st.session_state.messages_rod.append({"role": "user", "content": prompt})
                st.chat_message("user").markdown(prompt)
                try:
                    # 🚀 核心黑科技修复：使用 trust_env=False 彻底屏蔽本地代理配置！
                    custom_http_client = httpx.Client(trust_env=False)
                    
                    client = openai.OpenAI(
                        api_key=api_key, 
                        base_url="https://api.siliconflow.cn/v1",
                        http_client=custom_http_client  # 强制无代理直连
                    )
                    api_messages = [{"role": "system", "content": current_context}] + st.session_state.messages_rod
                    
                    response = client.chat.completions.create(
                        model=text_model, 
                        messages=api_messages,
                        temperature=0.1
                    )
                    ai_reply = response.choices[0].message.content
                    st.chat_message("assistant").markdown(ai_reply)
                    st.session_state.messages_rod.append({"role": "assistant", "content": ai_reply})
                except Exception as e: 
                    st.error(f"❌ API 错误: {e}")

# ==========================================
# 模块 B：智能功图图像识别系统 (视觉多模态)
# ==========================================
elif app_mode == "📈 智能功图图像识别":
    st.title("📈 智能功图图像识别系统")
    
    col_img, col_ai2 = st.columns([6, 4], gap="large")
    
    # --- B1. 左侧图像上传处理 ---
    with col_img:
        st.subheader("📸 上传功图 / 示功图照片")
        uploaded_file = st.file_uploader("支持 JPG, PNG 格式图片", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="当前待分析功图", use_container_width=True)
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        else:
            st.info("👆 请上传包含闭合曲线的油井功图照片，AI 将自动识别其工况特征（如供液不足、漏失、气锁等）。")
            base64_image = None

    # --- B2. 右侧视觉 Copilot ---
    with col_ai2:
        st.subheader(f"👁️ 功图诊断 Copilot")
        st.caption(f"当前视觉引擎：{vision_model.split('/')[-1]}")
        if st.button("🗑️ 清空功图对话", key="clear_card"): 
            st.session_state.messages_card = []
            st.rerun()
        
        chat_container2 = st.container(height=600)
        with chat_container2:
            for msg in st.session_state.messages_card:
                display_text = msg["content"]
                if isinstance(display_text, list): 
                    display_text = next((item["text"] for item in display_text if item["type"] == "text"), "[已发送分析图片]")
                st.chat_message(msg["role"]).markdown(display_text)

        if prompt := st.chat_input("例如：帮我分析这张功图，指出异常偏磨原因？"):
            if not api_key:
                st.error("请确认 API Key 是否填写！")
            elif not base64_image and not st.session_state.messages_card:
                st.warning("⚠️ 请先在左侧上传一张功图图片！")
            else:
                if base64_image and len(st.session_state.messages_card) == 0:
                    msg_content = [
                        {"type": "text", "text": f"作为资深采油工程师，请详细分析这张功图照片，指出曲线特征、可能的故障原因及处理建议：\n{prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                else:
                    msg_content = prompt

                st.session_state.messages_card.append({"role": "user", "content": msg_content})
                st.chat_message("user").markdown(prompt)
                
                try:
                    # 🚀 视觉模块同样装配 trust_env=False 引擎
                    custom_http_client = httpx.Client(trust_env=False)
                    
                    client = openai.OpenAI(
                        api_key=api_key, 
                        base_url="https://api.siliconflow.cn/v1",
                        http_client=custom_http_client  # 强制无代理直连
                    )
                    
                    response = client.chat.completions.create(
                        model=vision_model, 
                        messages=st.session_state.messages_card,
                        temperature=0.2
                    )
                    ai_reply = response.choices[0].message.content
                    st.chat_message("assistant").markdown(ai_reply)
                    st.session_state.messages_card.append({"role": "assistant", "content": ai_reply})
                except Exception as e:
                    st.error(f"❌ 视觉大模型调用失败，报错信息: {e}")
