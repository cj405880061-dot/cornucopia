import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io
import openai 
import math

# ==========================================
# 1. 网页全局设置与记忆初始化
# ==========================================
st.set_page_config(layout="wide", page_title="抽油杆智能化工具 (商业级)")
st.title("🛢️ 抽油杆多级管柱综合诊断系统 (Copilot 版)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. 侧边栏：专业级井况与工具配置面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 综合井况与管柱配置")
    
    st.markdown("### 1. 井筒与附加工具")
    tubing_size = st.selectbox("油管规格 (外径/内径 mm)", 
                               ["73.0 / 62.0 (2 7/8\")", "88.9 / 76.0 (3 1/2\")", "60.3 / 50.3 (2 3/8\")"])
    tubing_id = float(tubing_size.split('/')[1].split('(')[0].strip()) 
    
    # 井下工具与环境
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        use_anchor = st.checkbox("🔧 设油管锚", value=True, help="消除管柱弹性伸缩，增加杆柱有效载荷")
    with col_t2:
        use_echo = st.checkbox("📡 设回音标", help="在顶部增加回音标质量点")
        
    fluid_density = st.number_input("混合液密度 (kg/m³)", value=980, step=10)
    centralizer_density = st.slider("扶正器密度 (个/100m)", 0, 10, 2, help="防偏磨但增加重量与阻力")

    st.markdown("### 2. 抽油机与泵参数")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        stroke_length = st.number_input("冲程 (m)", 1.0, 8.0, 3.0, 0.1)
    with col_p2:
        stroke_rate = st.number_input("冲次 (次/min)", 1, 15, 6, 1)
    pump_diameter = st.selectbox("泵径 (mm)", [32, 38, 44, 57, 70], index=2)
    
    st.markdown("### 3. 多级抽油杆组合设计 (从下往上)")
    st.caption("💡 提示：可双击表格直接修改数值，或在最下方点击 ➕ 新增杆段")
    # 默认三级组合配杆表
    default_rod_data = pd.DataFrame({
        "段号": ["底部段 (接泵)", "中部段", "顶部段 (接悬点)"],
        "外径(mm)": [19.0, 22.0, 25.0],
        "长度(m)": [500.0, 500.0, 500.0],
        "钢级": ["D", "D", "H"]
    })
    
    rod_df = st.data_editor(
        default_rod_data, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    api_key = st.text_input("🔑 硅基流动 API Key", type="password")
    btn_calc = st.button("🚀 运行多级管柱综合校核", type="primary", use_container_width=True)
    if st.button("🗑️ 清空 AI 记忆"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. 核心计算大脑 (分段微元法)
# ==========================================
# 预备变量
result_df = pd.DataFrame()
total_depth = rod_df["长度(m)"].sum() if not rod_df.empty else 1500
min_safety = 99
current_context = ""
fig = go.Figure()

if btn_calc and not rod_df.empty:
    g = 9.81
    rho_steel = 7850
    r_p = (pump_diameter / 2) / 1000
    A_p = 3.14159 * (r_p ** 2)
    
    # 动态载荷加速度参数
    omega = 2 * 3.14159 * stroke_rate / 60
    a_max = (stroke_length / 2) * (omega ** 2)
    
    # 液柱载荷 (油管锚定系数)
    anchor_factor = 1.0 if use_anchor else 0.85 
    W_f = A_p * total_depth * fluid_density * g * anchor_factor
    
    cumulative_weight = 0
    cumulative_dynamic = 0
    section_results = []
    
    # 从下往上（反向遍历表）逐级递推计算
    rod_sections = rod_df.iloc[::-1].to_dict('records')
    
    for idx, row in enumerate(rod_sections):
        d_mm = max(float(row["外径(mm)"]), 1) # 防止填 0 报错
        L_m = float(row["长度(m)"])
        grade = row["钢级"]
        
        A_r = 3.14159 * ((d_mm / 2) / 1000) ** 2
        
        # 本段自重与扶正器重量
        W_self = A_r * L_m * rho_steel * g
        W_cent = (L_m / 100) * centralizer_density * 2 * g 
        W_section = W_self + W_cent
        
        # 水力阻力 (环空越小阻力越大)
        clearance = max(tubing_id - d_mm, 1)
        hydraulic_drag = (1 / clearance) * 50 * L_m 
        
        # 惯性动载荷
        F_d_section = W_section * (a_max / g)
        
        # 累加下方所有重量和动载荷
        cumulative_weight += W_section
        cumulative_dynamic += F_d_section
        
        # 计算该段【最顶部】的最大静载荷 + 动载荷
        max_load_here = cumulative_weight + cumulative_dynamic + W_f + hydraulic_drag
        if use_echo and idx == len(rod_sections) - 1: # 顶部加回音标质量
            max_load_here += 5 * g * (1 + a_max/g)
            
        stress_here = (max_load_here / A_r) / 1000000 # MPa
        
        # 许用应力判断
        yield_map = {"C": 413, "D": 586, "K": 413, "H": 792}
        allowable = yield_map.get(grade[0].upper(), 586) # 取首字母防错
        safety_here = allowable / stress_here if stress_here > 0 else 99
        
        section_results.append({
            "段落位置": f"从下往上 第{idx+1}段",
            "外径(mm)": d_mm,
            "长度(m)": L_m,
            "本段累积载荷(kN)": round(max_load_here / 1000, 2),
            "顶部最大应力(MPa)": round(stress_here, 2),
            "许用(MPa)": allowable,
            "安全系数": round(safety_here, 2)
        })

    # 将结果反转回从上往下的视觉顺序
    result_df = pd.DataFrame(section_results[::-1])
    min_safety = result_df["安全系数"].min()

    # ==========================================
    # 4. 智能化 3D 渲染 (带粗细阶梯的真实管柱)
    # ==========================================
    current_z = 0
    visual_exaggeration = 8 # 半径视觉放大 8 倍以便看清粗细变化

    for i, row in result_df.iterrows():
        L_m = row["长度(m)"]
        d_mm = row["外径(mm)"]
        stress = row["顶部最大应力(MPa)"]
        
        z_raw = np.linspace(current_z, current_z + L_m, 20)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z_raw)
        
        # 视觉半径
        r_render = ((d_mm / 2) / 1000) * visual_exaggeration
        x_grid = r_render * np.cos(theta_grid)
        y_grid = r_render * np.sin(theta_grid)
        
        # 给这一段赋予对应的应力颜色
        color_grid = np.full(z_grid.shape, stress)
        
        fig.add_trace(go.Surface(
            x=x_grid, y=y_grid, z=z_grid,
            surfacecolor=color_grid, colorscale='Jet',
            cmin=0, cmax=800, showscale=(i==0), # 只显示一次图例
            colorbar=dict(title="应力(MPa)", x=-0.1) if i==0 else None,
            showlegend=False, name=f"外径 {d_mm}mm"
        ))
        current_z += L_m

    # 添加井下工具标记 (3D 视觉元素)
    if use_anchor:
        # 在最底部画一个红色的圆环代表油管锚
        z_anchor = np.linspace(total_depth - 2, total_depth, 5)
        theta_a, z_a_grid = np.meshgrid(theta, z_anchor)
        r_anchor = r_render * 2.5
        x_a = r_anchor * np.cos(theta_a)
        y_a = r_anchor * np.sin(theta_a)
        fig.add_trace(go.Surface(x=x_a, y=y_a, z=z_a_grid, colorscale=[[0, 'red'], [1, 'red']], showscale=False, name="油管锚"))

    fig.update_layout(
        height=600, margin=dict(l=0, r=0, b=0, t=0), 
        scene=dict(
            zaxis_autorange="reversed",
            xaxis_title='-X-', yaxis_title='-Y-', zaxis_title='深度(m)',
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=3) 
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    )

    # 构造 AI 提示词矩阵
    context_table = result_df.to_markdown(index=False)
    current_context = f"""
    【当前系统参数】总井深 {total_depth}m, 冲程 {stroke_length}m, 冲次 {stroke_rate}次/min。油管内径 {tubing_id}mm。
    井下工具：安装油管锚={use_anchor}, 回音标={use_echo}, 扶正器={centralizer_density}个/100m。
    【各级杆柱力学计算结果】全井最低安全系数为 {min_safety}。以下是各段详细数据（从上到下）：
    {context_table}
    请作为油气田力学专家，分析以上组合段哪一段最容易发生断脱或偏磨？如果安全系数低于1.2，请直接建议我修改哪一段的外径或钢级。
    """

# ==========================================
# 5. 排版：左右 Copilot 分屏
# ==========================================
col_main, col_ai = st.columns([7, 3], gap="large")

with col_main:
    if not result_df.empty:
        st.subheader("📊 多级杆柱综合受力校核结果")
        st.dataframe(result_df, use_container_width=True)
        
        if min_safety < 1.2:
            st.error(f"🚨 危险告警：管柱中存在安全系数低于 1.2 的弱点（最低为 {min_safety}），建议向右侧 AI 提问优化方案！")
        else:
            st.success(f"✅ 校验通过：全井最低安全系数为 {min_safety}，结构安全。")

        st.subheader("🧊 多级阶梯管柱 3D 应力热力图 (粗细按比例映射)")
        st.plotly_chart(fig, use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False)
        st.download_button("📥 导出多级校核报告 (Excel)", data=buffer.getvalue(), file_name="多级配杆报告.xlsx")
    else:
        st.info("👈 请在左侧配置多级杆柱组合，并点击【运行多级管柱综合校核】")

# ==========================================
# 6. 右侧：全天候工程 Copilot
# ==========================================
with col_ai:
    st.subheader("💬 多级组合优化 Copilot")
    chat_container = st.container(height=650)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("向 AI 提问 (例：底部段安全系数太低，我该加粗还是换材质？)"):
        if not api_key:
            st.error("⚠️ 请先在左侧输入 API Key！")
        elif not current_context:
            st.warning("⚠️ 请先点击左侧运行校核，生成计算数据后再提问！")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    try:
                        client = openai.OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
                        api_messages = [{"role": "system", "content": current_context}] + st.session_state.messages
                        response = client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=api_messages)
                        ai_reply = response.choices[0].message.content
                        message_placeholder.markdown(ai_reply)
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    except Exception as e:
                        st.error(f"❌ 错误: {e}")
