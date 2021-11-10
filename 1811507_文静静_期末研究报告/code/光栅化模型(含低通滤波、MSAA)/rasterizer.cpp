// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include <sys/time.h>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;

struct timeval tv_begin, tv_end;

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// 检查当前点是否在三角形内
static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Eigen::Vector2f p;
	p << x, y;

	Eigen::Vector2f AB = _v[1].head(2) - _v[0].head(2);
	Eigen::Vector2f BC = _v[2].head(2) - _v[1].head(2);
	Eigen::Vector2f CA = _v[0].head(2) - _v[2].head(2);

	Eigen::Vector2f AP = p - _v[0].head(2);
	Eigen::Vector2f BP = p - _v[1].head(2);
	Eigen::Vector2f CP = p - _v[2].head(2);
	
	// 判断每个z坐标是否统一
	return AB[0] * AP[1] - AB[1] * AP[0] > 0 
		&& BC[0] * BP[1] - BC[1] * BP[0] > 0
		&& CA[0] * CP[1] - CA[1] * CP[0] > 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}
// 画图函数
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

void fft(int width, int height, Vector3f** fxRealTwo, Vector3f** fxImagTwo, Vector3f** RealTwo)
{
    cout << width << " " << height << endl;
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            fxRealTwo[v][u] = {0, 0, 0};
            fxImagTwo[v][u] = {0, 0, 0};
        }
    }
	for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int i = 0; i < height; i++)
                {
					float w = 2 * MY_PI * u * i / height + 2 * MY_PI * v * j / width;
                    fxRealTwo[v][u] += RealTwo[j][i] * cos(w);
                    fxImagTwo[v][u] -= RealTwo[j][i] * sin(w);
                }
            }
        }
    }
}

void ifft(int width, int height, Vector3f** ResultReal, Vector3f** ResultImage, Vector3f** ifxRealTwo, Vector3f** ifxImageTwo)
{
    cout << width << " " << height << endl;
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            ifxRealTwo[v][u] = {0, 0, 0};
            ifxImageTwo[v][u] = {0, 0, 0};
        }
    }
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            for (int j =0; j < width; j++)
            {
                for (int i = 0; i < height; i++)
                {
					float w = 2 * MY_PI * u * i / height + 2 * MY_PI * v * j / width;
                    ifxRealTwo[v][u] += ResultReal[j][i] * cos(w) - ResultImage[j][i] * sin(w);
                    ifxImageTwo[v][u] += ResultImage[j][i] * cos(w) + ResultReal[j][i] * sin(w);
                }
            }
            ifxRealTwo[v][u] /= (width * height);
            ifxImageTwo[v][u] /= (width * height);
        }
    }
}


//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    // auto v = t.toVector4();

	// 低通滤波

    float min_x = std::min(v[0][0], std::min(v[1][0], v[2][0])) - 1;
    float max_x = std::max(v[0][0], std::max(v[1][0], v[2][0])) + 1;
	float min_y = std::min(v[0][1], std::min(v[1][1], v[2][1])) - 1;
	float max_y = std::max(v[0][1], std::max(v[1][1], v[2][1])) + 1;

    min_x = (int)std::floor(min_x);  // 向下取整
	max_x = (int)std::ceil(max_x);  // 向上取整
	min_y = (int)std::floor(min_y);
	max_y = (int)std::ceil(max_y);

    int len_x = max_x - min_x + 1;
    int len_y = max_y - min_y + 1;

    Vector3f** RealTwo = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) RealTwo[i] = new Vector3f[len_y];

    for (int x = min_x; x <= max_x; x++) 
	{
		for (int y = min_y; y <= max_y; y++) 
        {
            if (insideTriangle((float)x + 0.5, (float)y + 0.5, t.v)) RealTwo[int(x - min_x)][int(y- min_y)] = t.getColor();
            else RealTwo[int(x - min_x)][int(y- min_y)] = {255, 255, 255};
        }
    }

    Vector3f** fxRealTwo = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) fxRealTwo[i] = new Vector3f[len_y];

    Vector3f** fxImagTwo = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) fxImagTwo[i] = new Vector3f[len_y];

    // cout << "bounding box 初始化完毕" << endl;

    // 对bounding box做FFT
    fft(len_x, len_y, fxRealTwo, fxImagTwo, RealTwo);
    // cout << "bounding box fft完毕" << endl;

    // 对卷积盒做FFT
    Vector3f** filter = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) filter[i] = new Vector3f[len_x];
    for(int i = 0; i < len_x; i++)
        for(int j = 0; j < len_x; j++)
            filter[i][j] = {0, 0, 0};

    for(int i = (len_x) / 2 - 1; i <= (len_x) / 2 + 1; i++)
        for(int j = (len_x) / 2 - 1; j <= (len_x) / 2 + 1; j++)
            filter[i][j] = {(float)1 / 3, (float)1 / 3, (float)1 / 3};
    
    Vector3f** FilterReal = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) FilterReal[i] = new Vector3f[len_x];

    Vector3f** FilterImage = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) FilterImage[i] = new Vector3f[len_x];

    // cout << "卷积盒初始化完毕" << endl;

    fft(len_x, len_x, FilterReal, FilterImage, filter);
    // cout << "卷积盒fft完毕" << endl;

    // 进行频域上的乘积
    Vector3f** ResultReal = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) ResultReal[i] = new Vector3f[len_y];

    Vector3f** ResultImage = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) ResultImage[i] = new Vector3f[len_y];

    // cout << "乘积初始化完毕" << endl;

    for(int j = 0; j < len_y; j++)  // 遍历左行
    {
        for(int y = 0; y < len_x; y++)  // 遍历右行
        {
            for(int x = 0; x < len_x; x++)  // 遍历列
            {
                ResultReal[x][j] = {
                                        fxRealTwo[x][j][0] * FilterReal[y][x][0] + fxImagTwo[x][j][0] * FilterImage[y][x][0],
                                        fxRealTwo[x][j][1] * FilterReal[y][x][1] + fxImagTwo[x][j][1] * FilterImage[y][x][1],
                                        fxRealTwo[x][j][2] * FilterReal[y][x][2] + fxImagTwo[x][j][2] * FilterImage[y][x][2]  
                                    };
                ResultImage[x][j] = {
                                        fxRealTwo[x][j][0] * FilterImage[y][x][0] + fxRealTwo[x][j][0] * FilterImage[y][x][0],
                                        fxRealTwo[x][j][1] * FilterImage[y][x][1] + fxRealTwo[x][j][1] * FilterImage[y][x][1],
                                        fxRealTwo[x][j][2] * FilterImage[y][x][2] + fxRealTwo[x][j][2] * FilterImage[y][x][2],
                                    };
            }
        }
    }

    // cout << "乘积完毕" << endl;

    // 对结果再做逆FFT
    Vector3f** ifxRealTwo = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) ifxRealTwo[i] = new Vector3f[len_y];

    Vector3f** ifxImagTwo = new Vector3f*[len_x];
    for(int i = 0; i < len_x; i++) ifxImagTwo[i] = new Vector3f[len_y];

    ifft(len_x, len_y, fxRealTwo, fxImagTwo, ifxRealTwo, ifxImagTwo);

    // cout << "逆傅里叶变换完毕" << endl;

	// // bounding box
	// float min_x = std::min(v[0][0], std::min(v[1][0], v[2][0]));
    // float max_x = std::max(v[0][0], std::max(v[1][0], v[2][0]));
	// float min_y = std::min(v[0][1], std::min(v[1][1], v[2][1]));
	// float max_y = std::max(v[0][1], std::max(v[1][1], v[2][1]));

	// min_x = (int)std::floor(min_x);
	// max_x = (int)std::ceil(max_x);
	// min_y = (int)std::floor(min_y);
	// max_y = (int)std::ceil(max_y);

	//MSAA 4X
	gettimeofday(&tv_begin, NULL);
	// 格子里的细分四个小点坐标
	std::vector<Eigen::Vector2f> pos
	{
		{0.25,0.25},
		{0.75,0.25},
		{0.25,0.75},
		{0.75,0.75},
	};
	// #pragma omp parallel for num_threads(4)
	for (int x = min_x; x <= max_x; x++) {
		for (int y = min_y; y <= max_y; y++) {
			// 记录最小深度
			float minDepth = FLT_MAX;
			// 四个小点中落入三角形中的点的个数
			int count = 0;
			// 对四个小点坐标进行判断
			#pragma omp parallel for num_threads(4)
			for (int i = 0; i < 4; i++) {
				// 小点是否在三角形内
				if (insideTriangle((float)x + pos[i][0], (float)y + pos[i][1], t.v)) {
					// 如果在，对深度z进行插值
					auto tup = computeBarycentric2D((float)x + pos[i][0], (float)y + pos[i][1], t.v);
					float alpha;
					float beta;
					float gamma;
					std::tie(alpha, beta, gamma) = tup;
					float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
					float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
					z_interpolated *= w_reciprocal;
					minDepth = std::min(minDepth, z_interpolated);
					count++;
				}
			}
			if (count != 0) {
				if (depth_buf[get_index(x, y)] > minDepth) {
					Vector3f color = t.getColor() * count / 4.0;
					Vector3f point(3);
					point << (float)x, (float)y, minDepth;
					// 替换深度
					depth_buf[get_index(x, y)] = minDepth;
					// 修改颜色
					set_pixel(point, color);
				}
			}
		}
	}
	gettimeofday(&tv_end, NULL);
	cout << "MSAA runing time = " << (float)(tv_begin.tv_sec - tv_end.tv_sec) * 1000.0 + (float)(tv_begin.tv_usec - tv_end.tv_usec) / 1000.0 << " ms" << endl;
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on