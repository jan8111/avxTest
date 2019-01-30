#include <stdio.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h> //AVX: -mavx
void avx_add(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
        vz[i] = _mm256_add_ps(vx[i], vy[i]);
    }
}

void avx_sub(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
        vz[i] = _mm256_sub_ps(vx[i], vy[i]);
    }
}

void avx_mul(const size_t n, float *x, float *y, float *z)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vz = (__m256 *)z;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    for(size_t i=0; i<end; ++i) {
        vz[i] = _mm256_mul_ps(vx[i], vy[i]);
    }
}

float avx_dot(const size_t n, float *x, float *y)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    __m256 vsum = {0};
    for(size_t i=0; i<end; ++i) {
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx[i], vy[i]));
    }
    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, vsum);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

float avx_euclidean_distance(const size_t n, float *x, float *y)
{
    static const size_t single_size = 8;
    const size_t end = n / single_size;
    __m256 *vx = (__m256 *)x;
    __m256 *vy = (__m256 *)y;
    __m256 vsub = {0};
    __m256 vsum = {0};
    for(size_t i=0; i<end; ++i) {
        vsub = _mm256_sub_ps(vx[i], vy[i]);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vsub, vsub));
    }
    __attribute__((aligned(32))) float t[8] = {0};
    _mm256_store_ps(t, vsum);
    return sqrt(t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]);
}


int testAvx() {
    float op1[8] = {2.2, 3.3, 4.4, 5.5, 5.5, 6.6, 7.7, 8.8};
    float op2[8] = {1.1, 2.2, 3.3, 4.4, 6.6, 7.7, 8.8, 9.9};
    float result[8];

    __m256 a = _mm256_loadu_ps(op1);
    __m256 b = _mm256_loadu_ps(op2);

    __m256 c = _mm256_sub_ps(a, b);   // c = a + b

    //c= _mm256_sqrt_ps(c);


    // Store
    _mm256_storeu_ps(result, c);

    for (int i = 0; i < 8; ++i) {
        printf("0: %lf\n", result[i]);
    }


    return 0;
}

int main() {
    //return testAvx();

    float x[] =   {(float) -0.024801115, (float)0.038720768, (float)0.086813, (float)0.03311974, (float)0.085429244, (float)-0.028036201, (float)-0.058523264, (float)-0.035835315, (float)0.0058188755, (float)-0.088805, (float)0.051243354, (float)0.0664497, (float)-0.04957725, (float)0.060375273, (float)-0.031798664, (float)-0.020924738, (float)0.010025513, (float)0.08216557, (float)0.031920422, (float)0.0023475515, (float)-0.023040151, (float)-0.11996659, (float)-0.050626013, (float)-0.09328891, (float)0.07212394, (float)0.023227563, (float)0.015407613, (float)-0.0065376386, (float)-0.0058144373, (float)0.046676148, (float)-0.14388391, (float)0.031506293, (float)0.029297655, (float)0.039443247, (float)-0.081367664, (float)-0.049653437, (float)0.0075000543, (float)-0.024012724, (float)-0.0046496154, (float)0.02704902, (float)0.022884907, (float)-0.04248466, (float)0.082925096, (float)-0.0035800003, (float)0.07725551, (float)-0.07860754, (float)0.076533645, (float)-0.11322852, (float)0.098407835, (float)0.058850303, (float)0.0333945, (float)0.11340917, (float)0.028743552, (float)-0.018791221, (float)-0.074958324, (float)-0.078533135, (float)-0.018433984, (float)-0.049669914, (float)-0.08460461, (float)-0.044133104, (float)-0.022163017, (float)-0.02257651, (float)0.049511895, (float)0.025269521, (float)0.100343436, (float)-0.016993046, (float)-0.06969719, (float)-0.08380028, (float)0.07115783, (float)-0.018693544, (float)0.07187634, (float)0.053445887, (float)0.013584088, (float)-0.017546246, (float)0.09942537, (float)0.046834704, (float)-0.0540828, (float)-0.0019518827, (float)-0.060117245, (float)-0.038542297, (float)-0.023396278, (float)0.07388047, (float)0.049988847, (float)0.00830321, (float)-0.03306911, (float)0.15245876, (float)-0.007898241, (float)0.033389412, (float)0.03750066, (float)-0.13249, (float)-0.13887706, (float)0.016788471, (float)-0.0038689466, (float)0.018630603, (float)-0.0990878, (float)0.042614046, (float)0.067458436, (float)-0.090727136, (float)-0.063337125, (float)0.04291697, (float)0.017426549, (float)0.09391713, (float)-0.037337422, (float)0.042709593, (float)0.011527899, (float)-0.03654068, (float)-0.06352701, (float)-0.031458307, (float)0.06260137, (float)-0.030917114, (float)-0.11162788, (float)-0.04898517, (float)-0.022574328, (float)0.03101806, (float)0.0073403083, (float)-0.04923205, (float)0.046487268, (float)0.0022930498, (float)-0.060902398, (float)0.008219514, (float)-0.09219243, (float)-0.06323009, (float)-0.12178671, (float)-0.034841426, (float)-0.05051028, (float)-0.014079814, (float)0.13579515, (float)0.0025445335, (float)-0.027386634, (float)-0.093120195, (float)-0.09860266, (float)0.08952215, (float)0.030505957, (float)0.046467047, (float)0.0597738, (float)0.068143964, (float)0.08690209, (float)-0.013194446, (float)-0.012771715, (float)-0.04248632, (float)-0.119694464, (float)0.03360648, (float)0.12866607, (float)0.06257382, (float)0.020797076, (float)-0.04000298, (float)0.12629738, (float)-0.04152932, (float)-0.037919573, (float)0.07549148, (float)-0.06369511, (float)0.015914109, (float)0.050441086, (float)0.027152708, (float)-0.039297212, (float)-0.02739397, (float)-0.00042752468, (float)-0.0024931505, (float)-0.0690489, (float)0.02940156, (float)-0.039503302, (float)0.03772984, (float)0.123370856, (float)0.027273735, (float)-0.0342156, (float)0.09882541, (float)-0.06043247, (float)-0.028960435, (float)0.034015585, (float)-0.09765231, (float)0.036991693, (float)0.044959754, (float)-0.09165632, (float)0.12593672, (float)-0.059905093, (float)-0.12731335, (float)0.07972109, (float)0.09541552, (float)0.044321146, (float)0.071505114, (float)-0.087247185, (float)0.034137722, (float)-0.056810908, (float)-0.05500674, (float)0.0054505174, (float)0.027946651, (float)-0.022340419, (float)-0.11293082, (float)0.015168888, (float)0.027233794, (float)-0.10498687, (float)-0.022122629, (float)-0.121265255, (float)-0.12766078, (float)0.0034646518, (float)-0.07722861, (float)-0.020169826, (float)0.0474508, (float)0.124721296, (float)0.015804483, (float)-0.083342485, (float)0.09799023, (float)0.0035925428, (float)0.00088168983, (float)-0.046100855, (float)-0.04280031, (float)0.010003309, (float)0.037925594, (float)-0.011954168, (float)-0.014540111, (float)0.01056212, (float)-0.029468162, (float)-0.03762158, (float)-0.017979972, (float)0.06915202, (float)0.01795184, (float)0.11354568, (float)0.105401985, (float)0.052747842, (float)0.04182601, (float)0.117823236, (float)-0.011082346, (float)-0.1152671, (float)0.05372999, (float)-0.014360207, (float)0.06765081, (float)0.010574324, (float)-0.071952, (float)0.07353892, (float)-0.033396382, (float)0.016075924, (float)-0.093651585, (float)-0.040703457, (float)0.08694611, (float)0.009565011, (float)-0.06489668, (float)0.062190052, (float)-0.0993673, (float)0.026598722, (float)-0.047664724, (float)-0.05589548, (float)0.02783732, (float)0.05374115, (float)-0.021283122, (float)0.0710928, (float)-0.0007820477, (float)0.059128743, (float)0.021851666, (float)0.017776571, (float)0.07750599, (float)-0.025360728, (float)0.1072485, (float)-0.008884269, (float)0.05072604, (float)-0.008172802, (float)0.04905587};
    float y[] =   {(float) -0.021719309, (float)0.050662372, (float)0.10911601, (float)0.03134895, (float)0.09519336, (float)-0.013859788, (float)-0.064004466, (float)-0.022124901, (float)0.03609468, (float)-0.05684647, (float)0.06511711, (float)0.058676433, (float)-0.046344586, (float)0.07343955, (float)-0.047983952, (float)-0.01919267, (float)0.02821823, (float)0.04914833, (float)0.03843619, (float)0.006400015, (float)-0.03432539, (float)-0.08236731, (float)-0.07006538, (float)-0.08131084, (float)0.07343308, (float)0.0064516184, (float)0.031009318, (float)0.0043541095, (float)-0.020206122, (float)0.028278705, (float)-0.07141178, (float)0.07065001, (float)0.007560196, (float)0.043815494, (float)-0.04927457, (float)-0.06399594, (float)0.03471451, (float)-0.003663267, (float)-0.023614695, (float)0.018123537, (float)0.04452656, (float)-0.058316745, (float)0.104198016, (float)-0.007742824, (float)0.053413536, (float)-0.045783594, (float)0.05233654, (float)-0.12108797, (float)0.12134884, (float)0.014314361, (float)0.07146005, (float)0.12153324, (float)0.07504356, (float)-0.0047456147, (float)-0.06839572, (float)-0.005595758, (float)-0.046418376, (float)-0.0583378, (float)-0.09573703, (float)-0.030383006, (float)-0.017521532, (float)-0.010654581, (float)0.040607817, (float)0.028580658, (float)0.03854702, (float)-0.034824323, (float)-0.07071848, (float)-0.088885255, (float)0.06194034, (float)0.009225698, (float)0.032131188, (float)0.036582943, (float)-0.023686344, (float)-0.053315606, (float)0.14406793, (float)0.022921594, (float)-0.03546867, (float)0.003473239, (float)-0.035207585, (float)0.010727515, (float)-0.03880905, (float)0.09805274, (float)0.059999768, (float)0.045651957, (float)-0.0107534835, (float)0.15809712, (float)-0.014668846, (float)0.048415814, (float)-0.0068937573, (float)-0.13872069, (float)-0.17626853, (float)0.04964738, (float)0.014354682, (float)0.025953703, (float)-0.056750286, (float)0.05651478, (float)0.05175409, (float)-0.05168446, (float)-0.030726867, (float)0.044490673, (float)0.019029023, (float)0.0258647, (float)-0.047730826, (float)0.04218469, (float)0.021956397, (float)-0.06525275, (float)-0.08911032, (float)-0.025642846, (float)0.054617114, (float)-0.013240187, (float)-0.09417527, (float)-0.04803924, (float)-0.031496096, (float)0.007392329, (float)-0.022480713, (float)-0.102037705, (float)0.026802594, (float)0.011772158, (float)-0.05898956, (float)0.0008612835, (float)-0.0761228, (float)-0.050051656, (float)-0.124846, (float)-0.06759116, (float)-0.071313165, (float)-0.02368228, (float)0.14765176, (float)0.018466273, (float)0.015194992, (float)-0.093445905, (float)-0.12660591, (float)0.03679542, (float)0.030746546, (float)0.055119257, (float)0.016101453, (float)0.0776836, (float)0.053346742, (float)-0.013614828, (float)-0.012122053, (float)-0.038523264, (float)-0.11601203, (float)-0.037604965, (float)0.10552626, (float)0.0579111, (float)0.0075810314, (float)-0.048771013, (float)0.13723159, (float)-0.0061612152, (float)-0.042970743, (float)0.059056845, (float)-0.08039791, (float)-0.015156509, (float)0.047473945, (float)0.035085797, (float)-0.035683986, (float)-0.032820642, (float)0.055136956, (float)0.016838502, (float)-0.04157346, (float)0.029024485, (float)-0.031590108, (float)0.032199416, (float)0.117909096, (float)0.00652731, (float)-0.056634184, (float)0.10855007, (float)-0.08321013, (float)-0.03379484, (float)0.039005913, (float)-0.07774771, (float)0.043839518, (float)0.045406725, (float)-0.0551916, (float)0.10241995, (float)-0.07882677, (float)-0.08044429, (float)0.0887034, (float)0.10130471, (float)0.036345646, (float)0.05323464, (float)-0.062544286, (float)0.018265817, (float)-0.0883997, (float)-0.07023716, (float)0.010164453, (float)-0.0018563434, (float)-0.0038970544, (float)-0.117655404, (float)0.060815945, (float)0.0496492, (float)-0.13727677, (float)0.0057668393, (float)-0.12024734, (float)-0.111104, (float)0.031248579, (float)-0.09305189, (float)-0.010526674, (float)0.017392172, (float)0.12532905, (float)0.04075386, (float)-0.10089486, (float)0.13384192, (float)-0.054180212, (float)0.0033944892, (float)-0.054716036, (float)-0.02199779, (float)-0.010390262, (float)0.041651834, (float)-0.028733207, (float)-0.03759505, (float)-0.009732947, (float)-0.010085503, (float)-0.002880071, (float)-0.033434037, (float)0.04888168, (float)-0.028763665, (float)0.099440366, (float)0.10356592, (float)0.07976008, (float)0.03592001, (float)0.15590623, (float)0.009350726, (float)-0.11244601, (float)0.047290545, (float)-0.03998565, (float)0.038328685, (float)-0.0428141, (float)-0.07261956, (float)-0.001334597, (float)-0.04241572, (float)0.0063615516, (float)-0.04779747, (float)-0.075738646, (float)0.07064096, (float)0.0058616977, (float)-0.10652438, (float)0.05384755, (float)-0.060486075, (float)0.01195863, (float)-0.040037062, (float)-0.047631685, (float)0.059155606, (float)0.05689161, (float)0.0066673215, (float)0.07158362, (float)-0.030322611, (float)0.06748566, (float)0.05366618, (float)0.03835174, (float)0.015947232, (float)-0.012847544, (float)0.0902552, (float)0.012389427, (float)0.08062854, (float)-0.036182933, (float)0.06777828};

    int length = sizeof(x)/sizeof(float);

    float distance1 = avx_euclidean_distance(256, x, y);
    printf("%f",distance1);

    return 0;
}
