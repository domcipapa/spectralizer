#include <string.h>
#include <assert.h>
#include <raylib.h>
#include <math.h>
#include <complex.h>
#include <rlgl.h>

#define FFT_SIZE (1<<13)
#define GLSL_VERSION 330

#define Float_Complex float complex
#define cfromreal(re) (re)
#define cfromimag(im) ((im) * I)
#define mulcc(a, b) ((a) * (b))
#define addcc(a, b) ((a) + (b))
#define subcc(a, b) ((a) - (b))

// window related
int window_width = 1600;
int window_height = 800;
char *window_title = "Audio Spectrum Visualizer in C";
int target_fps = 144;

// fft related
Float_Complex out_raw[FFT_SIZE];
float in_raw[FFT_SIZE];
float in_win[FFT_SIZE];
float out_log[FFT_SIZE];
float out_smooth[FFT_SIZE];
float out_smear[FFT_SIZE];

static void fft_push(float frame) {
    memmove(in_raw, in_raw + 1, (FFT_SIZE - 1) * sizeof(in_raw[0]));
    in_raw[FFT_SIZE - 1] = frame;
}

static void callback(void *bufferData, unsigned int frames) {
    float (*fs)[2] = bufferData;
    for (size_t i = 0; i < frames; ++i) {
        fft_push(fs[i][0]);
    }
}

static void fft(float in[], size_t stride, Float_Complex out[], size_t n) {
    assert(n > 0);

    if (n == 1) {
        out[0] = cfromreal(in[0]);
        return;
    }

    fft(in, stride * 2, out, n / 2);
    fft(in + stride, stride * 2, out + n / 2, n / 2);

    for (size_t k = 0; k < n / 2; ++k) {
        float t = (float) k / n;
        Float_Complex v = mulcc(cexp(cfromimag(-2 * PI * t)), out[k + n / 2]);
        Float_Complex e = out[k];
        out[k] = addcc(e, v);
        out[k + n / 2] = subcc(e, v);
    }
}

static inline float amp(Float_Complex z) {
    float a = crealf(z);
    float b = cimagf(z);
    return logf(a * a + b * b);
}

static size_t fft_analyze(float dt) {
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        float t = (float) i / (FFT_SIZE - 1);
        float hann = 0.5 - 0.5 * cosf(2 * PI * t);
        in_win[i] = in_raw[i] * hann;
    }

    fft(in_win, 1, out_raw, FFT_SIZE);

    float step = 1.06f;
    float lowf = 1.0f;
    size_t m = 0;
    float max_amp = 1.0f;

    for (float f = lowf; (size_t) f < FFT_SIZE / 2; f = ceilf(f * step)) {
        float f1 = ceilf(f * step);
        float a = 0.0f;
        for (size_t q = (size_t) f; q < FFT_SIZE / 2 && q < (size_t) f1; ++q) {
            float b = amp(out_raw[q]);
            if (b > a) a = b;
        }
        if (max_amp < a) max_amp = a;
        out_log[m++] = a;
    }

    for (size_t i = 0; i < m; ++i) {
        out_log[i] /= max_amp;
    }

    for (size_t i = 0; i < m; ++i) {
        float smoothness = 8;
        out_smooth[i] += (out_log[i] - out_smooth[i]) * smoothness * dt;
        float smearness = 3;
        out_smear[i] += (out_smooth[i] - out_smear[i]) * smearness * dt;
    }

    return m;
}

Shader circle;
int circle_radius_location;
int circle_power_location;
static void fft_render(Rectangle boundary, size_t m) {
    // Width of a single bar
    float cell_width = boundary.width / m;

    // Color related
    float saturation = 0.75f;
    float value = 1.0f;

    // Draw LINES
    for (size_t i = 0; i < m; ++i) {
        float t = out_smooth[i];
        float hue = (float) i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);

        Vector2 startPos = {
                boundary.x + i * cell_width + cell_width / 2,
                boundary.y + boundary.height - boundary.height * 2 / 3 * t,
        };
        Vector2 endPos = {
                boundary.x + i * cell_width + cell_width / 2,
                boundary.y + boundary.height,
        };

        float thick = cell_width / 3 * sqrtf(t);
        DrawLineEx(startPos, endPos, thick, color);
    }

    // Load texture
    Texture2D texture = { rlGetTextureIdDefault(), 1, 1, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };

    // Draw SMEARS
    SetShaderValue(circle, circle_radius_location, (float[1]) { 0.3f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(circle, circle_power_location, (float[1]) { 0.3f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(circle);

    for (size_t i = 0; i < m; ++i) {
        float start = out_smear[i];
        float end = out_smooth[i];
        float hue = (float) i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);

        Vector2 startPos = {
                boundary.x + i * cell_width + cell_width / 2,
                boundary.y + boundary.height - boundary.height * 2 / 3 * start,
        };

        Vector2 endPos = {
                boundary.x + i * cell_width + cell_width / 2,
                boundary.y + boundary.height - boundary.height * 2 / 3 * end,
        };

        float radius = cell_width * 3 * sqrtf(end);
        Vector2 origin = { 0 };

        if (endPos.y >= startPos.y) {
            Rectangle dest = {
                    .x = startPos.x - radius / 2,
                    .y = startPos.y,
                    .width = radius,
                    .height = endPos.y - startPos.y,
            };

            Rectangle source = { 0, 0, 1, 0.5 };
            DrawTexturePro(texture, source, dest, origin, 0, color);
        } else {
            Rectangle dest = {
                    .x = endPos.x - radius / 2,
                    .y = endPos.y,
                    .width = radius,
                    .height = startPos.y - endPos.y,
            };

            Rectangle source = { 0, 0.5, 1, 0.5 };
            DrawTexturePro(texture, source, dest, origin, 0, color);
        }
    }
    EndShaderMode();

    // Draw CIRCLES
    SetShaderValue(circle, circle_radius_location, (float[1]) { 0.07f }, SHADER_UNIFORM_FLOAT);
    SetShaderValue(circle, circle_power_location, (float[1]) { 5.0f }, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(circle);

    for (size_t i = 0; i < m; ++i) {
        float t = out_smooth[i];
        float hue = (float) i / m;
        Color color = ColorFromHSV(hue * 360, saturation, value);

        Vector2 center = {
                boundary.x + i * cell_width + cell_width / 2,
                boundary.y + boundary.height - boundary.height * 2 / 3 * t,
        };

        float radius = cell_width * 6 * sqrtf(t);
        Vector2 position = {
                .x = center.x - radius,
                .y = center.y - radius,
        };

        DrawTextureEx(texture, position, 0, 2 * radius, color);
    }

    EndShaderMode();
}

int main(int argc, char **argv) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
    InitWindow(window_width, window_height, window_title);
    SetTargetFPS(target_fps);

    InitAudioDevice();
    Music music = LoadMusicStream("../legion.mp3");
    AttachAudioStreamProcessor(music.stream, callback);
    PlayMusicStream(music);

    circle = LoadShader(0, TextFormat("../resources/shaders/glsl%d/circle.fs", GLSL_VERSION));
    circle_radius_location = GetShaderLocation(circle, "radius");
    circle_power_location = GetShaderLocation(circle, "power");

    while (!WindowShouldClose()) {
        BeginDrawing(); {
            ClearBackground(BLACK);
            int w = GetScreenWidth();
            int h = GetScreenHeight();

            UpdateMusicStream(music);
            size_t m = fft_analyze(GetFrameTime());

            Rectangle preview_boundary = {
                    .x = 0,
                    .y = 0,
                    .width = w,
                    .height = h,
            };

            DrawFPS(10, 10);
            fft_render(preview_boundary, m);
        } EndDrawing();
    }

    CloseAudioDevice();
    CloseWindow();
    return 0;
}
