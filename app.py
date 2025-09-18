import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata, interp2d, RegularGridInterpolator, Rbf
from scipy.ndimage import gaussian_filter
import io
import base64

# 페이지 설정
st.set_page_config(
    page_title="ISO Polar Plot Visualization", 
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정 (matplotlib - 크로스섹션용)
plt.rcParams['font.family'] = ['DejaVu Sans',  'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_and_process_data(uploaded_file):
    """CSV 파일을 로드하고 처리하는 함수"""
    try:
        df = pd.read_csv(uploaded_file)

        # 데이터 구조 확인
        if 'Theta' not in df.columns:
            st.error("CSV 파일에 'Theta' 컬럼이 없습니다.")
            return None

        # Phi 컬럼들 (숫자로 된 컬럼명) 찾기
        phi_columns = [col for col in df.columns if col != 'Theta' and col.replace('.', '').replace('-','').isdigit()]
        phi_values = [float(col) for col in phi_columns]

        # 데이터를 long format으로 변환
        df_long = df.melt(id_vars=['Theta'], 
                         value_vars=phi_columns, 
                         var_name='Phi', 
                         value_name='Luminance')

        df_long['Phi'] = df_long['Phi'].astype(float)
        df_long['Theta'] = df_long['Theta'].astype(float)
        df_long['Luminance'] = df_long['Luminance'].astype(float)

        return df_long, phi_values

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def create_plotly_smooth_polar_plot(df_long, vmin, vmax, cmap='Jet', resolution=300):
    """Plotly로 부드러운 극좌표 플롯 생성 - 숫자 라벨 제거 및 interpolation 개선"""
    
    # 데이터 준비
    theta_vals = np.sort(df_long['Theta'].unique())
    phi_vals = np.sort(df_long['Phi'].unique())
    
    # 고해상도 보간을 위한 격자 생성
    theta_interp = np.linspace(0, theta_vals.max(), resolution//2)
    phi_interp = np.linspace(0, 360, resolution)
    
    # 원본 데이터를 2D 배열로 재구성
    df_pivot = df_long.pivot(index='Theta', columns='Phi', values='Luminance')
    
    # 강화된 보간
    try:
        # 먼저 결측값을 최근접 이웃으로 채움
        df_pivot_filled = df_pivot.fillna(method='ffill').fillna(method='bfill')
        
        # Regular grid interpolator 사용
        f_interp = RegularGridInterpolator(
            (df_pivot.index.values, df_pivot.columns.values),
            df_pivot_filled.values,
            method='linear',
            bounds_error=False,
            fill_value=np.nanmean(df_pivot_filled.values)
        )
        
        theta_grid, phi_grid = np.meshgrid(theta_interp, phi_interp, indexing='ij')
        points = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)
        luminance_interp = f_interp(points).reshape(theta_grid.shape)
        
        # 가우시안 필터로 부드럽게 처리
        luminance_interp = gaussian_filter(luminance_interp, sigma=1.0)
        
    except Exception as e:
        st.warning(f"고급 보간 실패, 기본 보간 사용: {str(e)}")
        # 기본 보간 방법
        x = []
        y = []
        z = []
        for _, row in df_long.iterrows():
            x.append(row['Theta'])
            y.append(row['Phi'])
            z.append(row['Luminance'])
        
        theta_grid, phi_grid = np.meshgrid(theta_interp, phi_interp, indexing='ij')
        luminance_interp = griddata(
            (x, y), z, (theta_grid, phi_grid), 
            method='linear', fill_value=np.nanmean(z)
        )
        # NaN 값을 평균값으로 채움
        luminance_interp = np.nan_to_num(luminance_interp, nan=np.nanmean(luminance_interp))
    
    # Plotly 컬러맵 설정
    plotly_colorscales = {
        'jet': 'Jet',
        'viridis': 'Viridis', 
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'hot': 'Hot',
        'cool': 'RdBu',
        'rainbow': 'Rainbow',
        'turbo': 'Turbo'
    }
    plotly_cmap = plotly_colorscales.get(cmap.lower(), 'Jet')
    
    # Plotly figure 생성
    fig = go.Figure()
    
    # 컨투어 플롯 추가 (더 조밀한 포인트로)
    fig.add_trace(go.Scatterpolar(
        r=theta_grid.flatten(),
        theta=phi_grid.flatten(),
        mode='markers',
        marker=dict(
            size=2,  # 포인트 크기 축소
            color=luminance_interp.flatten(),
            colorscale=plotly_cmap,
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="Luminance", side="right"),
                tickmode="linear",
                tick0=vmin,
                dtick=(vmax-vmin)/5
            )
        ),
        name='ISO Data',
        hovertemplate='Theta: %{r:.1f}°<br>Phi: %{theta:.1f}°<br>Luminance: %{marker.color:.2f}<extra></extra>'
    ))
    
    #Box 데이터
    # box_data = {
    #     'P_A+': [(10.74, 201.6),(10.74, -21.6),(12.71, 38.6),(12.71, 141.4)],
    #     'P_A': [(22.02, 205.8), (22.02, -25.8), (27.24, 45), (27.24, 135)],
    #     'D_A+': [(55.03, -2.8), (55.13, 5.62),(45.28, 8), (45.07, 356)  ],
    #     'D_A': [(60.13, 354.2), (60.53, 11.9), (42.45, 23.45), (40.61, 348.1) ]
    # }
    # # Box 데이터 (10도 상향)
    box_data = {
        'P_A+': [(11.60, 149.2),(11.60, 30.8),(20.29, 61.5),(20.29, 118.5)],
        'P_A': [(20.0, 180.0), (20, 0), (34.31, 57.8), (34.31, 122)],
        'D_A+': [(55.07, 4.2), (55.68, 12.8),(46.44, 18), (45.16, 6)  ],
        'D_A': [(60.00, 0), (61.29, 18.4), (45.53, 34.5), (40, 0) ]
    }
    
    # Box 플롯 추가
    for box_name, box_coords in box_data.items():
        r = [coord[0] for coord in box_coords]
        theta = [(coord[1] + 180) % 360 for coord in box_coords]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]],  # 첫 번째 점으로 돌아가서 닫힌 도형을 만듦
            theta=theta + [theta[0]],  # 첫 번째 점으로 돌아가서 닫힌 도형을 만듦
            mode='lines',
            line=dict(color='black', width=1),
            name=box_name,
            hoverinfo='skip'
        ))


    



    # 레이아웃 설정 - 숫자 라벨 제거
    fig.update_layout(
        title=f'ISO Luminance Distribution (Polar)<br>(Range: {vmin:.1f} - {vmax:.1f})',
        polar=dict(
            radialaxis=dict(
                visible=False,  # 반지름 축 숨김
                range=[0, theta_vals.max()]
            ),
            angularaxis=dict(
                visible=False,  # 각도 축 숨김
                direction='clockwise',
                rotation=0  # rotation을 0으로 변경
            )
        ),
        width=700,
        height=700,
        font=dict(size=12),
        showlegend=False
    )
    
    return fig



def create_plotly_cartesian_plot(df_long, vmin, vmax, cmap='Jet', resolution=300, box_data=None):
    """Plotly로 직교좌표계 플롯 생성 - Audi 10도 상향"""
    
    # 극좌표를 직교좌표로 변환
    theta_rad = np.radians(df_long['Theta'])
    phi_rad = np.radians(df_long['Phi'])
    r_norm = df_long['Theta'] / df_long['Theta'].max()
    
    x = r_norm * np.cos(phi_rad)
    y = r_norm * np.sin(phi_rad)
    
    # 고해상도 격자 생성
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 강화된 보간
    try:
        # RBF 보간을 먼저 시도
        rbf = Rbf(x, y, df_long['Luminance'], function='multiquadric', smooth=0.05)
        zi = rbf(xi_grid, yi_grid)
        
        # 가우시안 필터로 부드럽게 처리
        zi = gaussian_filter(zi, sigma=1.5)
        
    except Exception as e:
        st.warning(f"RBF 보간 실패, griddata 사용: {str(e)}")
        try:
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='linear')
            # NaN 값을 최근접 이웃으로 채움
            zi_nearest = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='nearest')
            zi = np.where(np.isnan(zi), zi_nearest, zi)
            
        except Exception as e2:
            st.warning(f"griddata 보간도 실패, 최근접 이웃 사용: {str(e2)}")
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='nearest')
    
    # 원형 마스크 적용
    mask = xi_grid**2 + yi_grid**2 <= 1.02**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # Plotly 컬러맵 설정
    plotly_colorscales = {
        'jet': 'Jet',
        'viridis': 'Viridis',
        'plasma': 'Plasma', 
        'inferno': 'Inferno',
        'hot': 'Hot',
        'cool': 'RdBu',
        'rainbow': 'Rainbow',
        'turbo': 'Turbo'
    }
    plotly_cmap = plotly_colorscales.get(cmap.lower(), 'Jet')
    
    # Plotly figure 생성
    fig = go.Figure()
    
    # 히트맵 추가
    fig.add_trace(go.Heatmap(
        x=xi,
        y=yi,
        z=zi_masked,
        colorscale=plotly_cmap,
        zmin=vmin,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Luminance", side="right")
        ),
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Luminance: %{z:.2f}<extra></extra>'
    ))
    
    # 원형 경계 추가
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)
    
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='white', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Box 플롯 추가
    if box_data is not None:
        for box_name, box_coords in box_data.items():
            x_box = [coord[0] for coord in box_coords]
            y_box = [coord[1] for coord in box_coords]
            fig.add_trace(go.Scatter(
                x=x_box + [x_box[0]],  # 첫 번째 점으로 돌아가서 닫힌 도형을 만듦
                y=y_box + [y_box[0]],  # 첫 번째 점으로 돌아가서 닫힌 도형을 만듦
                mode='lines',
                line=dict(color='black', width=1),
                name=box_name,
                hoverinfo='skip'
            ))
    
    # 레이아웃 설정 - 숫자 라벨 제거
    fig.update_layout(
        title=f'ISO Luminance Distribution (Cartesian)<br>(Range: {vmin:.1f} - {vmax:.1f})',
        xaxis=dict(
            range=[-1.3, 1.3],
            showgrid=False,
            zeroline=False,
            showticklabels=False,  # 숫자 라벨 제거
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-1.3, 1.3], 
            showgrid=False,
            zeroline=False,
            showticklabels=False  # 숫자 라벨 제거
        ),
        width=700,
        height=700,
        plot_bgcolor='black',
        font=dict(size=12),
        showlegend=False
    )

    return fig



def create_plotly_cartesian_plot(df_long, vmin, vmax, cmap='Jet', resolution=300):
    """Plotly로 직교좌표계 플롯 생성 - 숫자 라벨 제거"""
    
    # 극좌표를 직교좌표로 변환
    theta_rad = np.radians(df_long['Theta'])
    phi_rad = np.radians(df_long['Phi'])
    r_norm = df_long['Theta'] / df_long['Theta'].max()
    
    x = r_norm * np.cos(phi_rad)
    y = r_norm * np.sin(phi_rad)
    
    # 고해상도 격자 생성
    xi = np.linspace(-1.1, 1.1, resolution)
    yi = np.linspace(-1.1, 1.1, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 강화된 보간
    try:
        # RBF 보간을 먼저 시도
        rbf = Rbf(x, y, df_long['Luminance'], function='multiquadric', smooth=0.05)
        zi = rbf(xi_grid, yi_grid)
        
        # 가우시안 필터로 부드럽게 처리
        zi = gaussian_filter(zi, sigma=1.5)
        
    except Exception as e:
        st.warning(f"RBF 보간 실패, griddata 사용: {str(e)}")
        try:
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='linear')
            # NaN 값을 최근접 이웃으로 채움
            zi_nearest = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='nearest')
            zi = np.where(np.isnan(zi), zi_nearest, zi)
            
        except Exception as e2:
            st.warning(f"griddata 보간도 실패, 최근접 이웃 사용: {str(e2)}")
            zi = griddata((x, y), df_long['Luminance'], (xi_grid, yi_grid), method='nearest')
    
    # 원형 마스크 적용
    mask = xi_grid**2 + yi_grid**2 <= 1.02**2
    zi_masked = np.where(mask, zi, np.nan)
    
    # Plotly 컬러맵 설정
    plotly_colorscales = {
        'jet': 'Jet',
        'viridis': 'Viridis',
        'plasma': 'Plasma', 
        'inferno': 'Inferno',
        'hot': 'Hot',
        'cool': 'RdBu',
        'rainbow': 'Rainbow',
        'turbo': 'Turbo'
    }
    plotly_cmap = plotly_colorscales.get(cmap.lower(), 'Jet')
    
    # Plotly figure 생성
    fig = go.Figure()
    
    # 히트맵 추가
    fig.add_trace(go.Heatmap(
        x=xi,
        y=yi,
        z=zi_masked,
        colorscale=plotly_cmap,
        zmin=vmin,
        zmax=vmax,
        showscale=True,
        colorbar=dict(
            title=dict(text="Luminance", side="right")
        ),
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Luminance: %{z:.2f}<extra></extra>'
    ))
    
    # 원형 경계 추가
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta_circle)
    y_circle = np.sin(theta_circle)
    
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='white', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 레이아웃 설정 - 숫자 라벨 제거
    fig.update_layout(
        title=f'ISO Luminance Distribution (Cartesian)<br>(Range: {vmin:.1f} - {vmax:.1f})',
        xaxis=dict(
            range=[-1.3, 1.3],
            showgrid=False,
            zeroline=False,
            showticklabels=False,  # 숫자 라벨 제거
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[-1.3, 1.3], 
            showgrid=False,
            zeroline=False,
            showticklabels=False  # 숫자 라벨 제거
        ),
        width=700,
        height=700,
        plot_bgcolor='black',
        font=dict(size=12),
        showlegend=False
    )
    
    return fig
    
def create_plotly_cross_section(df_long, cross_direction):
    """Plotly로 크로스섹션 플롯 생성 - 하나의 그래프에 양방향 표시"""
    
    try:
        # Plotly figure 생성 (단일 그래프)
        fig = go.Figure()
        
        if cross_direction == "가로 (0°-180°)":
            # 0도와 180도 방향의 휘도 프로파일
            df_0 = df_long[df_long['Phi'] == 0].copy()
            df_180 = df_long[df_long['Phi'] == 180].copy()
            
            # 데이터가 없는 경우 가장 가까운 값 찾기
            if df_0.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 0).abs().argsort()[:1]].values[0]
                df_0 = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"Phi=0° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
            
            if df_180.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 180).abs().argsort()[:1]].values[0]
                df_180 = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"Phi=180° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
            
            df_0 = df_0.sort_values('Theta')
            df_180 = df_180.sort_values('Theta')
            
            # 0도 방향 프로파일
            fig.add_trace(
                go.Scatter(x=df_0['Theta'], y=df_0['Luminance'],
                          mode='lines+markers', name='0° direction (우측)',
                          line=dict(color='red', width=3),
                          marker=dict(size=8, symbol='circle'),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<br>Direction: 0° (우측)<extra></extra>')
            )
            
            # 180도 방향 프로파일
            fig.add_trace(
                go.Scatter(x=df_180['Theta'], y=df_180['Luminance'],
                          mode='lines+markers', name='180° direction (좌측)',
                          line=dict(color='blue', width=3),
                          marker=dict(size=8, symbol='square'),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<br>Direction: 180° (좌측)<extra></extra>')
            )
            
            title = "가로 방향 휘도 프로파일 (0°-180°)"
            
        else:  # 세로 (90°-270°)
            # 90도와 270도 방향의 휘도 프로파일
            df_90 = df_long[df_long['Phi'] == 90].copy()
            df_270 = df_long[df_long['Phi'] == 270].copy()
            
            # 데이터가 없는 경우 가장 가까운 값 찾기
            if df_90.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 90).abs().argsort()[:1]].values[0]
                df_90 = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"Phi=90° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
            
            if df_270.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 270).abs().argsort()[:1]].values[0]
                df_270 = df_long[df_long['Phi'] == closest_phi].copy()
                st.info(f"Phi=270° 데이터가 없어 가장 가까운 {closest_phi}° 사용")
            
            df_90 = df_90.sort_values('Theta')
            df_270 = df_270.sort_values('Theta')
            
            # 90도 방향 프로파일
            fig.add_trace(
                go.Scatter(x=df_90['Theta'], y=df_90['Luminance'],
                          mode='lines+markers', name='90° direction (상단)',
                          line=dict(color='green', width=3),
                          marker=dict(size=8, symbol='circle'),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<br>Direction: 90° (상단)<extra></extra>')
            )
            
            # 270도 방향 프로파일
            fig.add_trace(
                go.Scatter(x=df_270['Theta'], y=df_270['Luminance'],
                          mode='lines+markers', name='270° direction (하단)',
                          line=dict(color='magenta', width=3),
                          marker=dict(size=8, symbol='square'),
                          hovertemplate='Theta: %{x}°<br>Luminance: %{y:.2f}<br>Direction: 270° (하단)<extra></extra>')
            )
            
            title = "세로 방향 휘도 프로파일 (90°-270°)"
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Theta (degrees)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Luminance",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            font=dict(size=12),
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Plotly 크로스섹션 생성 중 오류: {str(e)}")
        # 에러 메시지가 포함된 기본 figure 반환
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"크로스섹션 생성 실패<br>{str(e)}",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Error in Cross-section Generation",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig

def create_cross_section_csv_data(df_long, cross_direction):
    """크로스섹션 데이터를 CSV 형태로 생성"""
    
    try:
        if cross_direction == "가로 (0°-180°)":
            # 0도와 180도 방향의 휘도 프로파일
            df_0 = df_long[df_long['Phi'] == 0].copy()
            df_180 = df_long[df_long['Phi'] == 180].copy()
            
            # 데이터가 없는 경우 가장 가까운 값 찾기
            if df_0.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 0).abs().argsort()[:1]].values[0]
                df_0 = df_long[df_long['Phi'] == closest_phi].copy()
            
            if df_180.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 180).abs().argsort()[:1]].values[0]
                df_180 = df_long[df_long['Phi'] == closest_phi].copy()
            
            df_0 = df_0.sort_values('Theta')
            df_180 = df_180.sort_values('Theta')
            
            # 데이터를 병합하여 CSV 형태로 만들기
            csv_data = pd.DataFrame({
                'Theta': df_0['Theta'],
                'Luminance_0deg': df_0['Luminance'],
            })
            
            # 180도 데이터 추가 (theta 값이 다를 수 있으므로 병합)
            df_180_renamed = df_180[['Theta', 'Luminance']].rename(columns={'Luminance': 'Luminance_180deg'})
            csv_data = pd.merge(csv_data, df_180_renamed, on='Theta', how='outer')
            
        else:  # 세로 (90°-270°)
            # 90도와 270도 방향의 휘도 프로파일
            df_90 = df_long[df_long['Phi'] == 90].copy()
            df_270 = df_long[df_long['Phi'] == 270].copy()
            
            # 데이터가 없는 경우 가장 가까운 값 찾기
            if df_90.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 90).abs().argsort()[:1]].values[0]
                df_90 = df_long[df_long['Phi'] == closest_phi].copy()
            
            if df_270.empty:
                closest_phi = df_long['Phi'].iloc[(df_long['Phi'] - 270).abs().argsort()[:1]].values[0]
                df_270 = df_long[df_long['Phi'] == closest_phi].copy()
            
            df_90 = df_90.sort_values('Theta')
            df_270 = df_270.sort_values('Theta')
            
            # 데이터를 병합하여 CSV 형태로 만들기
            csv_data = pd.DataFrame({
                'Theta': df_90['Theta'],
                'Luminance_90deg': df_90['Luminance'],
            })
            
            # 270도 데이터 추가
            df_270_renamed = df_270[['Theta', 'Luminance']].rename(columns={'Luminance': 'Luminance_270deg'})
            csv_data = pd.merge(csv_data, df_270_renamed, on='Theta', how='outer')
        
        # NaN 값을 빈 문자열로 채우기
        csv_data = csv_data.fillna('')
        
        return csv_data
        
    except Exception as e:
        st.error(f"CSV 데이터 생성 중 오류: {str(e)}")
        # 에러 시 빈 DataFrame 반환
        return pd.DataFrame({'Error': [f'CSV 생성 실패: {str(e)}']})

def save_plotly_as_html(fig, filename):
    """Plotly 그래프를 HTML로 저장"""
    try:
        html_str = fig.to_html(include_plotlyjs='cdn')
        return html_str.encode()
    except Exception as e:
        st.error(f"HTML 저장 실패: {str(e)}")
        error_html = f"""
        <html>
        <head><title>Error</title></head>
        <body>
        <h1>Plot Generation Error</h1>
        <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        return error_html.encode()

def save_matplotlib_as_png(fig):
    """Matplotlib 그래프를 PNG 바이트로 저장"""
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        return img_buffer.getvalue()
    except Exception as e:
        st.error(f"PNG 저장 실패: {str(e)}")
        img_buffer = io.BytesIO()
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Image Generation Error\n{str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Error")
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()

def main():
    st.title("📊 Enhanced ISO Polar Plot Visualization")
    st.markdown("CSV 파일을 업로드하여 ISO(광학 강도 분포) polar plot과 크로스섹션을 생성합니다.")

    # 사이드바
    st.sidebar.header("설정")

    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 선택", 
        type=['csv'],
        help="Theta 컬럼과 각도별 Phi 값들이 포함된 CSV 파일을 업로드하세요."
    )

    if uploaded_file is not None:
        # 데이터 로드
        result = load_and_process_data(uploaded_file)

        if result is not None:
            df_long, phi_values = result

            # 데이터 정보 표시
            st.sidebar.success("✅ 데이터 로드 성공!")
            st.sidebar.write(f"**데이터 포인트:** {len(df_long)}")
            st.sidebar.write(f"**Theta 범위:** {df_long['Theta'].min()}° - {df_long['Theta'].max()}°")
            st.sidebar.write(f"**Phi 범위:** {df_long['Phi'].min()}° - {df_long['Phi'].max()}°")

            # 데이터 범위 정보
            data_min = float(df_long['Luminance'].min())
            data_max = float(df_long['Luminance'].max())
            st.sidebar.write(f"**Luminance 범위:** {data_min:.2f} - {data_max:.2f}")

            st.sidebar.divider()

            # 컬러바 범위 설정
            st.sidebar.subheader("🎨 컬러바 설정")

            colorbar_mode = st.sidebar.radio(
                "컬러바 범위 모드",
                ["자동 (데이터 범위)", "수동 설정"],
                help="자동: 데이터의 최소/최대값 사용, 수동: 직접 범위 설정"
            )

            if colorbar_mode == "자동 (데이터 범위)":
                vmin, vmax = data_min, data_max
                st.sidebar.info(f"자동 범위: {vmin:.2f} ~ {vmax:.2f}")
            else:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    vmin = st.number_input("최소값", value=data_min, step=0.1, format="%.2f")
                with col2:
                    vmax = st.number_input("최대값", value=data_max, step=0.1, format="%.2f")
                
                if vmin >= vmax:
                    st.sidebar.error("최소값은 최대값보다 작아야 합니다!")
                    vmin, vmax = data_min, data_max

            # 컬러맵 선택
            colormap_options = ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Hot', 'Cool', 'Rainbow', 'Turbo']
            selected_cmap = st.sidebar.selectbox("컬러맵", colormap_options, index=0)

            # 해상도 설정
            resolution = st.sidebar.slider("해상도", min_value=100, max_value=500, value=300, step=50)

            st.sidebar.divider()

            # 크로스섹션 설정
            st.sidebar.subheader("✂️ 크로스섹션 설정")
            
            cross_direction = st.sidebar.selectbox(
                "크로스섹션 방향",
                ["가로 (0°-180°)", "세로 (90°-270°)"],
                help="가로: 0°-180° 방향으로 자른 휘도 프로파일, 세로: 90°-270° 방향으로 자른 휘도 프로파일"
            )
            
            # 선택된 방향에 따른 정보 표시
            if cross_direction == "가로 (0°-180°)":
                st.sidebar.info("💡 가로 방향: 0°(우측)과 180°(좌측) 방향의 휘도 프로파일을 하나의 그래프에 표시합니다.")
            else:
                st.sidebar.info("💡 세로 방향: 90°(상단)과 270°(하단) 방향의 휘도 프로파일을 하나의 그래프에 표시합니다.")

            # 메인 컨텐츠
            tab1, tab2, tab3 = st.tabs(["📊 Main Plots", "✂️ Cross-sections", "📊 Data Info"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Plotly Polar Plot (숫자 라벨 제거)")
                    try:
                        fig_polar = create_plotly_smooth_polar_plot(df_long, vmin, vmax, selected_cmap, resolution)
                        st.plotly_chart(fig_polar, use_container_width=True)
                        
                        html_data = save_plotly_as_html(fig_polar, "polar_plot.html")
                        st.download_button(
                            label="🌐 Polar Plot HTML 다운로드",
                            data=html_data,
                            file_name="iso_polar_plot.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Polar plot 생성 실패: {str(e)}")
                
                with col2:
                    st.subheader("Plotly Cartesian Plot (숫자 라벨 제거)")
                    try:
                        fig_cartesian = create_plotly_cartesian_plot(df_long, vmin, vmax, selected_cmap, resolution)
                        st.plotly_chart(fig_cartesian, use_container_width=True)
                        
                        html_data_cart = save_plotly_as_html(fig_cartesian, "cartesian_plot.html")
                        st.download_button(
                            label="🌐 Cartesian Plot HTML 다운로드",
                            data=html_data_cart,
                            file_name="iso_cartesian_plot.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.error(f"Cartesian plot 생성 실패: {str(e)}")

            with tab2:
                st.subheader(f"크로스섹션: {cross_direction}")
                
                # Plotly 크로스섹션 (인터랙티브)
                try:
                    fig_cross_plotly = create_plotly_cross_section(df_long, cross_direction)
                    st.plotly_chart(fig_cross_plotly, use_container_width=True)
                    
                    # 다운로드 버튼들을 나란히 배치
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        html_data_cross = save_plotly_as_html(fig_cross_plotly, "cross_section.html")
                        st.download_button(
                            label="🌐 Cross-section HTML 다운로드",
                            data=html_data_cross,
                            file_name="cross_section.html",
                            mime="text/html"
                        )
                    
                    with col2:
                        # CSV 다운로드 버튼
                        csv_cross_data = create_cross_section_csv_data(df_long, cross_direction)
                        csv_cross_bytes = csv_cross_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Cross-section CSV 다운로드",
                            data=csv_cross_bytes,
                            file_name=f"cross_section_{cross_direction.replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('-', '_')}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"인터랙티브 크로스섹션 생성 실패: {str(e)}")
                
                st.divider()
                
                # Matplotlib 크로스섹션 (PNG 저장용)
                st.subheader("크로스섹션 (PNG 저장용)")
                try:
                    fig_cross_mpl = create_cross_section_plots(df_long, cross_direction)
                    st.pyplot(fig_cross_mpl)
                    
                    png_data = save_matplotlib_as_png(fig_cross_mpl)
                    st.download_button(
                        label="🖼️ 크로스섹션 PNG 다운로드",
                        data=png_data,
                        file_name=f"cross_section_{cross_direction.replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg').replace('-', '_')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"PNG 크로스섹션 생성 실패: {str(e)}")

            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("데이터 미리보기")
                    st.dataframe(df_long.head(10))
                
                with col2:
                    st.subheader("통계 정보")
                    st.write("**Luminance 통계:**")
                    stats_df = pd.DataFrame({
                        '통계': ['최소값', '최대값', '평균', '표준편차', '중간값'],
                        '값': [
                            f"{df_long['Luminance'].min():.2f}",
                            f"{df_long['Luminance'].max():.2f}",
                            f"{df_long['Luminance'].mean():.2f}",
                            f"{df_long['Luminance'].std():.2f}",
                            f"{df_long['Luminance'].median():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True)
                
                csv_data = df_long.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 전처리된 데이터 다운로드 (CSV)",
                    data=csv_data,
                    file_name="processed_iso_data.csv",
                    mime="text/csv"
                )

    else:
        st.info("👆 사이드바에서 CSV 파일을 업로드하세요.")
        st.subheader("📋 주요 개선사항")
        st.markdown("""
        ### ✨ **최신 업데이트:**
        
        #### 1. **숫자 라벨 제거**
        - ISO plot에서 모든 숫자 라벨과 각도 표시 제거
        - 깔끔한 시각화로 데이터 패턴에 집중
        
        #### 2. **크로스섹션 분석 개선**
        - **가로 방향 (0°-180°)**: 우측(0°)과 좌측(180°) 휘도 프로파일을 하나의 그래프에 표시
        - **세로 방향 (90°-270°)**: 상단(90°)과 하단(270°) 휘도 프로파일을 하나의 그래프에 표시
        - 양방향 데이터를 동시 비교하여 대칭성 분석 용이
        - HTML, PNG, CSV 형태로 다운로드 지원
        
        #### 3. **강화된 Interpolation**
        - 가우시안 필터링으로 부드러운 표면 생성
        - NaN 값 처리 개선으로 빈 공간 제거
        - RBF 보간과 griddata 조합으로 최적화
        
        #### 4. **다양한 내보내기 형식**
        - 📊 메인 플롯: HTML 형태로 저장
        - ✂️ 크로스섹션: HTML (인터랙티브) + PNG (고품질 이미지) + CSV (데이터)
        - 📊 데이터: 전처리된 CSV 파일
        - 양방향 휘도 프로파일 데이터를 하나의 CSV에 포함
        
        #### 5. **향상된 사용성**
        - 직관적인 크로스섹션 방향 선택 (가로/세로)
        - 하나의 그래프에서 양방향 데이터 동시 비교
        - 에러 상황에서 자동 복구 기능
        - 더 세밀한 해상도 조절
        
        ### 📁 **CSV 파일 형식**: 
        - 첫 번째 컬럼: `Theta` (각도 값)
        - 나머지 컬럼: 각 Phi 각도 (0, 10, 20, ..., 360)
        
        ### 💾 **크로스섹션 CSV 출력 형식**:
        - **가로 방향**: Theta, Luminance_0deg, Luminance_180deg
        - **세로 방향**: Theta, Luminance_90deg, Luminance_270deg
        """)

if __name__ == "__main__":
    main()
