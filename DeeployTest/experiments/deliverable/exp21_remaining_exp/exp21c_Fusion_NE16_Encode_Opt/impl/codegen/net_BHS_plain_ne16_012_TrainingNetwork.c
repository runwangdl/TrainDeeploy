#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "TrainingNetwork.h"

int8_t *DeeployNetwork_MEMORYARENA_L1;
int8_t *DeeployNetwork_MEMORYARENA_L2;
float32_t *DeeployNetwork_input_0;
int64_t *DeeployNetwork_input_1;
float32_t *DeeployNetwork_input_2;
float32_t *DeeployNetwork_input_3;
float32_t *DeeployNetwork_input_4;
float32_t *DeeployNetwork_input_5;
float32_t *DeeployNetwork_input_6;
float32_t *DeeployNetwork_input_7;
float32_t *DeeployNetwork_input_8;
float32_t *DeeployNetwork_input_9;
float32_t *DeeployNetwork_input_10;
float32_t *DeeployNetwork_input_11;
int8_t *DeeployNetwork_input_12;
int8_t *DeeployNetwork_input_13;
int8_t *DeeployNetwork_input_14;
int8_t *DeeployNetwork_input_15;
int8_t *DeeployNetwork_input_16;
int32_t *DeeployNetwork_input_17;
int32_t *DeeployNetwork_input_18;
int32_t *DeeployNetwork_input_19;
int32_t *DeeployNetwork_input_20;
int32_t *DeeployNetwork_input_21;
float32_t *DeeployNetwork_input_22;
float32_t *DeeployNetwork_input_23;
float32_t *DeeployNetwork_output_0;
float32_t *DeeployNetwork_output_1;

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16bias__dual4[8] = {
    85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {16715, 28051, 19407, 24750, 20593, 26069, 21962, 15816};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {421, 251, 363, 284, 342, 270, 321, 445};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22[16] = {
    234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192, 219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {3887854, 4028793, 3998692, 4560022, 4527256, 4318277, 4195455, 4117846,
                                                                              3644722, 4679690, 3915633, 4146882, 4516702, 3937397, 4507614, 4631827};

static PI_L2 int32_t
    DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22[16] = {
        69, 67, 67, 59, 60, 62, 64, 66, 74, 58, 69, 65, 60, 69, 60, 58};

static PI_L2 int32_t DeeployNetwork_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37[16] = {
    187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670, 249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor[16] = {5779922, 6811814, 7569997, 7017406, 6391308, 6797053, 6348470, 6703570,
                                                                              7695533, 4777151, 6888298, 6076617, 7800891, 6162310, 4713482, 8093983};

static PI_L2 int32_t
    DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37[16] = {
        136, 115, 104, 112, 123, 116, 124, 117, 102, 165, 114, 129, 101, 128, 167, 97};

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16wenc__dual3[8] = {
    85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21[16] = {
    234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192, 219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36[16] = {
    187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670, 249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_x2__dual8_tensor[16] = {
    421, 251, 363, 284, 342, 270, 321, 445, 421, 251, 363, 284, 342, 270, 321, 445};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_mean__dual17_tensor[16] = {
    -0.0019563890527933836f,  -4.265925963409245e-05f,  0.0010741822188720107f,  -0.00022483340580947697f, -3.8470159779535607e-05f, -0.00020210926595609635f,
    0.00030828602029941976f,  0.00018801260739564896f,  -0.0019563890527933836f, -4.265925963409245e-05f,  0.0010741822188720107f,   -0.00022483340580947697f,
    -3.8470159779535607e-05f, -0.00020210926595609635f, 0.00030828602029941976f, 0.00018801260739564896f};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_var__dual18_tensor[16] = {
    10794.6962890625f, 1960.42578125f, 1352.2359619140625f, 1431.591064453125f, 4565.24951171875f, 1289.480712890625f, 1300.5797119140625f, 14838.5048828125f,
    10794.6962890625f, 1960.42578125f, 1352.2359619140625f, 1431.591064453125f, 4565.24951171875f, 1289.480712890625f, 1300.5797119140625f, 14838.5048828125f};

static PI_L2 int32_t
    DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16__dual26[16] = {
        69, 67, 67, 59, 60, 62, 64, 66, 74, 58, 69, 65, 60, 69, 60, 58};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_mean__dual32_tensor[32] = {
    -0.2444174885749817f,  0.21689638495445251f, 0.14622187614440918f, -1.0542712211608887f, -1.0498594045639038f, -0.22530169785022736f, -1.1160809993743896f,
    1.0315797328948975f,   1.0182175636291504f,  0.12470299005508423f, -1.3734524250030518f, -0.7460818886756897f, -0.670293390750885f,   -1.0082124471664429f,
    0.6750643253326416f,   0.8555498123168945f,  -0.2444174885749817f, 0.21689638495445251f, 0.14622187614440918f, -1.0542712211608887f,  -1.0498594045639038f,
    -0.22530169785022736f, -1.1160809993743896f, 1.0315797328948975f,  1.0182175636291504f,  0.12470299005508423f, -1.3734524250030518f,  -0.7460818886756897f,
    -0.670293390750885f,   -1.0082124471664429f, 0.6750643253326416f,  0.8555498123168945f};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_var__dual33_tensor[32] = {
    1.4679807424545288f, 1.4829299449920654f, 4.743453502655029f,  5.358211040496826f,  4.712860107421875f, 4.743685245513916f,  4.386803150177002f,
    2.8539552688598633f, 3.485712766647339f,  1.066118597984314f,  10.345640182495117f, 5.888366222381592f, 1.6831457614898682f, 3.925990581512451f,
    2.867671489715576f,  2.335334300994873f,  1.4679807424545288f, 1.4829299449920654f, 4.743453502655029f, 5.358211040496826f,  4.712860107421875f,
    4.743685245513916f,  4.386803150177002f,  2.8539552688598633f, 3.485712766647339f,  1.066118597984314f, 10.345640182495117f, 5.888366222381592f,
    1.6831457614898682f, 3.925990581512451f,  2.867671489715576f,  2.335334300994873f};

static PI_L2 int32_t
    DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16__dual41[16] = {
        136, 115, 104, 112, 123, 116, 124, 117, 102, 165, 114, 129, 101, 128, 167, 97};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_mean__dual47_tensor[32] = {
    -0.4543471932411194f,  0.033739298582077026f,  -0.16740328073501587f, -0.633358359336853f,     0.06192547827959061f,  -0.29306620359420776f,
    -0.18620505928993225f, -0.16867128014564514f,  0.24714651703834534f,  -0.09643757343292236f,   -0.2440318465232849f,  -0.32666322588920593f,
    0.3093155026435852f,   -0.48663511872291565f,  -0.43303781747817993f, -0.0018389797769486904f, -0.4543471932411194f,  0.033739298582077026f,
    -0.16740328073501587f, -0.633358359336853f,    0.06192547827959061f,  -0.29306620359420776f,   -0.18620505928993225f, -0.16867128014564514f,
    0.24714651703834534f,  -0.09643757343292236f,  -0.2440318465232849f,  -0.32666322588920593f,   0.3093155026435852f,   -0.48663511872291565f,
    -0.43303781747817993f, -0.0018389797769486904f};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_var__dual48_tensor[32] = {
    0.6995296478271484f,  0.35785973072052f,   1.5051546096801758f,  0.8668981790542603f, 0.4478089213371277f, 0.4923137128353119f, 1.185558557510376f,
    0.9423964023590088f,  1.2930549383163452f, 0.180169016122818f,   0.5536366105079651f, 0.6328718662261963f, 0.5637519955635071f, 0.537251353263855f,
    0.16963136196136475f, 0.5420557260513306f, 0.6995296478271484f,  0.35785973072052f,   1.5051546096801758f, 0.8668981790542603f, 0.4478089213371277f,
    0.4923137128353119f,  1.185558557510376f,  0.9423964023590088f,  1.2930549383163452f, 0.180169016122818f,  0.5536366105079651f, 0.6328718662261963f,
    0.5637519955635071f,  0.537251353263855f,  0.16963136196136475f, 0.5420557260513306f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    82, 84, 104, 115, 98, 79, 118, 183, 109, 113, 105, 71, 113, 104, 117, 79, 94, 89, 99, 103, 125, 123, 87, 130, 75, 81, 91, 133, 81, 172, 72, 91};

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor[32] = {
    11216862, 10953840, 8883199, 8006243, 9415140, 11637480, 7831187,  5033047, 8487970,  8172161,  8744505,  12902138, 8150436,  8865462, 7886326,  11680739,
    9828662,  10359709, 9262086, 8927959, 7369362, 7468230,  10595193, 7107887, 12231511, 11419024, 10140785, 6934845,  11424804, 5359367, 12871468, 10140571};

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_mean__dual54_tensor[64] = {
    -0.6443479657173157f, -0.8780460357666016f,  -0.9605752229690552f, -0.728754460811615f,    -1.379516839981079f,  0.36568009853363037f,
    -0.976334273815155f,  -0.8195656538009644f,  -0.3788541853427887f, -1.4433526992797852f,   -1.4827815294265747f, -0.8990912437438965f,
    -0.4861598014831543f, -1.4708205461502075f,  -0.7284005284309387f, 0.6005582213401794f,    1.5618020296096802f,  -1.0277910232543945f,
    -1.1303950548171997f, -1.396936058998108f,   -1.281982183456421f,  -0.4754037857055664f,   -1.0184197425842285f, -1.0318992137908936f,
    -0.5048392415046692f, -0.43774089217185974f, -1.4235376119613647f, -0.006140610668808222f, -0.9846276044845581f, -0.4856228530406952f,
    -0.5889110565185547f, -1.107672095298767f,   -0.6443479657173157f, -0.8780460357666016f,   -0.9605752229690552f, -0.728754460811615f,
    -1.379516839981079f,  0.36568009853363037f,  -0.976334273815155f,  -0.8195656538009644f,   -0.3788541853427887f, -1.4433526992797852f,
    -1.4827815294265747f, -0.8990912437438965f,  -0.4861598014831543f, -1.4708205461502075f,   -0.7284005284309387f, 0.6005582213401794f,
    1.5618020296096802f,  -1.0277910232543945f,  -1.1303950548171997f, -1.396936058998108f,    -1.281982183456421f,  -0.4754037857055664f,
    -1.0184197425842285f, -1.0318992137908936f,  -0.5048392415046692f, -0.43774089217185974f,  -1.4235376119613647f, -0.006140610668808222f,
    -0.9846276044845581f, -0.4856228530406952f,  -0.5889110565185547f, -1.107672095298767f};

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_var__dual55_tensor[64] = {
    0.2816118896007538f,  0.3946440517902374f,  0.4444875419139862f,  0.5381677150726318f,  0.43645036220550537f, 0.7069288492202759f,  0.7624120116233826f,
    0.35157787799835205f, 0.452341765165329f,   0.3499625027179718f,  0.6326277852058411f,  0.3337295353412628f,  0.26218539476394653f, 0.6095684766769409f,
    0.5870797634124756f,  0.5308420062065125f,  0.541131854057312f,   0.35709550976753235f, 0.3786107897758484f,  0.33237600326538086f, 0.607239305973053f,
    0.304384708404541f,   0.47302573919296265f, 0.47411835193634033f, 0.31873422861099243f, 0.40998321771621704f, 0.5107912421226501f,  0.35335618257522583f,
    0.2936612665653229f,  0.4064280688762665f,  0.3643214702606201f,  0.36573007702827454f, 0.2816118896007538f,  0.3946440517902374f,  0.4444875419139862f,
    0.5381677150726318f,  0.43645036220550537f, 0.7069288492202759f,  0.7624120116233826f,  0.35157787799835205f, 0.452341765165329f,   0.3499625027179718f,
    0.6326277852058411f,  0.3337295353412628f,  0.26218539476394653f, 0.6095684766769409f,  0.5870797634124756f,  0.5308420062065125f,  0.541131854057312f,
    0.35709550976753235f, 0.3786107897758484f,  0.33237600326538086f, 0.607239305973053f,   0.304384708404541f,   0.47302573919296265f, 0.47411835193634033f,
    0.31873422861099243f, 0.40998321771621704f, 0.5107912421226501f,  0.35335618257522583f, 0.2936612665653229f,  0.4064280688762665f,  0.3643214702606201f,
    0.36573007702827454f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    163, 133, 150, 149, 149, 142, 134, 125, 109, 159, 123, 127, 137, 133, 222, 115,
    134, 127, 142, 118, 139, 144, 123, 146, 106, 130, 128, 146, 108, 159, 155, 130};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor[32] = {15255979, 18601466, 16480003, 16604952, 16636081, 17501801, 18550852, 19833580,
                                                                              22784586, 15600707, 20171281, 19515017, 18112376, 18597217, 11157841, 21481128,
                                                                              18522609, 19594714, 17518029, 20954163, 17792839, 17273134, 20128746, 16975140,
                                                                              23468860, 19124083, 19414542, 16972130, 22860964, 15612485, 16032134, 19108972};

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_mean__dual61_tensor[64] = {
    0.42469096183776855f,  0.32596051692962646f,  0.10398848354816437f,  -0.3451862037181854f,  -0.36299625039100647f, -0.19927622377872467f,
    -0.446662962436676f,   -0.6463051438331604f,  -0.5271444320678711f,  -0.20434394478797913f, 0.4471416473388672f,   -0.15140807628631592f,
    -0.07134925574064255f, 0.40762627124786377f,  -0.06986096501350403f, 0.3484388291835785f,   -0.0795891210436821f,  -0.2456866353750229f,
    0.1434946060180664f,   0.034666508436203f,    0.5462859272956848f,   0.46823567152023315f,  -0.3238901197910309f,  -0.6293811798095703f,
    0.325870156288147f,    0.07531605660915375f,  0.19413888454437256f,  -0.15748722851276398f, 0.16247861087322235f,  -0.5395596623420715f,
    0.23268234729766846f,  -0.3300189971923828f,  0.42469096183776855f,  0.32596051692962646f,  0.10398848354816437f,  -0.3451862037181854f,
    -0.36299625039100647f, -0.19927622377872467f, -0.446662962436676f,   -0.6463051438331604f,  -0.5271444320678711f,  -0.20434394478797913f,
    0.4471416473388672f,   -0.15140807628631592f, -0.07134925574064255f, 0.40762627124786377f,  -0.06986096501350403f, 0.3484388291835785f,
    -0.0795891210436821f,  -0.2456866353750229f,  0.1434946060180664f,   0.034666508436203f,    0.5462859272956848f,   0.46823567152023315f,
    -0.3238901197910309f,  -0.6293811798095703f,  0.325870156288147f,    0.07531605660915375f,  0.19413888454437256f,  -0.15748722851276398f,
    0.16247861087322235f,  -0.5395596623420715f,  0.23268234729766846f,  -0.3300189971923828f};

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_var__dual62_tensor[64] = {
    0.1964213103055954f,  0.17257089912891388f, 0.14762769639492035f, 0.3156340420246124f,  0.2242252230644226f,  0.2599714398384094f,  0.13144634664058685f,
    0.21899321675300598f, 0.2966296672821045f,  0.2763214111328125f,  0.21101363003253937f, 0.1880793422460556f,  0.27006587386131287f, 0.1884487271308899f,
    0.19838258624076843f, 0.16746504604816437f, 0.2212221771478653f,  0.17600755393505096f, 0.19948957860469818f, 0.1521081030368805f,  0.17710629105567932f,
    0.2113693207502365f,  0.3071841895580292f,  0.31239357590675354f, 0.23397785425186157f, 0.19457979500293732f, 0.09693852812051773f, 0.241731658577919f,
    0.3287730813026428f,  0.19810181856155396f, 0.1785918027162552f,  0.23535661399364471f, 0.1964213103055954f,  0.17257089912891388f, 0.14762769639492035f,
    0.3156340420246124f,  0.2242252230644226f,  0.2599714398384094f,  0.13144634664058685f, 0.21899321675300598f, 0.2966296672821045f,  0.2763214111328125f,
    0.21101363003253937f, 0.1880793422460556f,  0.27006587386131287f, 0.1884487271308899f,  0.19838258624076843f, 0.16746504604816437f, 0.2212221771478653f,
    0.17600755393505096f, 0.19948957860469818f, 0.1521081030368805f,  0.17710629105567932f, 0.2113693207502365f,  0.3071841895580292f,  0.31239357590675354f,
    0.23397785425186157f, 0.19457979500293732f, 0.09693852812051773f, 0.241731658577919f,   0.3287730813026428f,  0.19810181856155396f, 0.1785918027162552f,
    0.23535661399364471f};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_cumByteOffset[1] = {0};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride[4] = {1884, 1884, 920, 920};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride[4] = {7536, 7536, 3680, 3680};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[4] = {4, 1, 4, 1};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo[4] = {157, 157, 77, 77};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[4] = {3, 3, 2, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[4] = {3, 3, 2, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 4};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[4] = {0, 33648, 1884, 35532};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[4] = {1988688, 1969848, 1977120, 1967920};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[4] = {22608, 3768, 11040, 1840};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[4] = {1884, 1884, 920, 920};

static PI_L2 int32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[4] = {33648, -31764, 33648, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[4] = {0, 134592, 7536, 142128};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[4] = {1925440, 1850080, 1879168, 1842368};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[4] = {90432, 15072, 44160, 7360};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[4] = {7536, 7536, 3680, 3680};

static PI_L2 int32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[4] = {134592, -127056, 134592, 0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out[2] = {9, 5};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[2] = {0, 2};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cumByteOffset[2] = {0, 100944};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cmd[2] = {1542736, 1497872};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_cumByteOffset[2] = {0, 0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cumByteOffset[2] = {0, 12528};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cmd[2] = {1323248, 1317680};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_cumByteOffset[1] = {
    0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_cumByteOffset[1] = {
    0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[24];
void *DeeployNetwork_outputs[2];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_x21c_hcache[23];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_x21c_hcache[39];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_x21c_hcache[39];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_x21c_hcache[71];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_x21c_hcache[1031];
static PI_L2 uint32_t DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_x21c_hcache[1031];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref;
} _pert_fcbias_fp32_mode1__dual71_tiling_closure_args_t;

static void _pert_fcbias_fp32_mode1__dual71_tiling_closure(void *_pert_fcbias_fp32_mode1__dual71_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual71_tiling_closure_args_t *args =
      (_pert_fcbias_fp32_mode1__dual71_tiling_closure_args_t *)_pert_fcbias_fp32_mode1__dual71_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcbias_fp32_mode1__dual71, Op: PerturbRademacherFixed, mode 1) -- QW exp8
  uint8_t pert_fcbias_fp32_mode1__dual71_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcbias_fp32_mode1__dual71_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcbias_fp32_mode1__dual71_chunk = (9 >> pert_fcbias_fp32_mode1__dual71_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcbias_fp32_mode1__dual71_chunk_start =
      (uint32_t)MIN(pert_fcbias_fp32_mode1__dual71_chunk * pert_fcbias_fp32_mode1__dual71_core_id, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode1__dual71_chunk_stop =
      (uint32_t)MIN(pert_fcbias_fp32_mode1__dual71_chunk_start + pert_fcbias_fp32_mode1__dual71_chunk, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode1__dual71_local_size = pert_fcbias_fp32_mode1__dual71_chunk_stop - pert_fcbias_fp32_mode1__dual71_chunk_start;
  uint32_t pert_fcbias_fp32_mode1__dual71_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_fcbias_fp32_mode1__dual71_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcbias_fp32_mode1__dual71_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref[pert_fcbias_fp32_mode1__dual71_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref)[pert_fcbias_fp32_mode1__dual71_chunk_start],
      pert_fcbias_fp32_mode1__dual71_seed, 1u, pert_fcbias_fp32_mode1__dual71_local_size, pert_fcbias_fp32_mode1__dual71_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref;
} _pert_fcbias_fp32_mode1__dual71_cluster_fork_args_t;

static void _pert_fcbias_fp32_mode1__dual71_cluster_fork(void *_pert_fcbias_fp32_mode1__dual71_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual71_cluster_fork_args_t *args =
      (_pert_fcbias_fp32_mode1__dual71_cluster_fork_args_t *)_pert_fcbias_fp32_mode1__dual71_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode1__dual71_tiling_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual71_tiling_closure_args =
      (_pert_fcbias_fp32_mode1__dual71_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref};

  // _pert_fcbias_fp32_mode1__dual71_tiling_closure CLOSURE CALL
  _pert_fcbias_fp32_mode1__dual71_tiling_closure(&DeeployNetwork__pert_fcbias_fp32_mode1__dual71_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr;
} _pert_fcbias_fp32_mode1__dual71_closure_args_t;

static void _pert_fcbias_fp32_mode1__dual71_closure(void *_pert_fcbias_fp32_mode1__dual71_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual71_closure_args_t *args = (_pert_fcbias_fp32_mode1__dual71_closure_args_t *)_pert_fcbias_fp32_mode1__dual71_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_input_23_ref =
      (void *)((char *)DeeployNetwork_input_23 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_fcbias_fp32_pert_p__dual70_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_input_23_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcbias_fp32_mode1__dual71_cluster_fork_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual71_cluster_fork_args =
        (_pert_fcbias_fp32_mode1__dual71_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcbias_fp32_mode1__dual71_cluster_fork, &DeeployNetwork__pert_fcbias_fp32_mode1__dual71_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_fcbias_fp32_pert_p__dual70_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr;
} _pert_fcbias_fp32_mode1__dual71_closure_L3_args_t;

static void _pert_fcbias_fp32_mode1__dual71_closure_L3(void *_pert_fcbias_fp32_mode1__dual71_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual71_closure_L3_args_t *args =
      (_pert_fcbias_fp32_mode1__dual71_closure_L3_args_t *)_pert_fcbias_fp32_mode1__dual71_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode1__dual71_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual71_closure_args = (_pert_fcbias_fp32_mode1__dual71_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr};

  // _pert_fcbias_fp32_mode1__dual71_closure CLOSURE CALL
  _pert_fcbias_fp32_mode1__dual71_closure(&DeeployNetwork__pert_fcbias_fp32_mode1__dual71_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref;
} _pert_fcbias_fp32_mode2__dual77_tiling_closure_args_t;

static void _pert_fcbias_fp32_mode2__dual77_tiling_closure(void *_pert_fcbias_fp32_mode2__dual77_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual77_tiling_closure_args_t *args =
      (_pert_fcbias_fp32_mode2__dual77_tiling_closure_args_t *)_pert_fcbias_fp32_mode2__dual77_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcbias_fp32_mode2__dual77, Op: PerturbRademacherFixed, mode 2) -- QW exp8
  uint8_t pert_fcbias_fp32_mode2__dual77_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcbias_fp32_mode2__dual77_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcbias_fp32_mode2__dual77_chunk = (9 >> pert_fcbias_fp32_mode2__dual77_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcbias_fp32_mode2__dual77_chunk_start =
      (uint32_t)MIN(pert_fcbias_fp32_mode2__dual77_chunk * pert_fcbias_fp32_mode2__dual77_core_id, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode2__dual77_chunk_stop =
      (uint32_t)MIN(pert_fcbias_fp32_mode2__dual77_chunk_start + pert_fcbias_fp32_mode2__dual77_chunk, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode2__dual77_local_size = pert_fcbias_fp32_mode2__dual77_chunk_stop - pert_fcbias_fp32_mode2__dual77_chunk_start;
  uint32_t pert_fcbias_fp32_mode2__dual77_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_fcbias_fp32_mode2__dual77_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcbias_fp32_mode2__dual77_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref[pert_fcbias_fp32_mode2__dual77_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref)[pert_fcbias_fp32_mode2__dual77_chunk_start],
      pert_fcbias_fp32_mode2__dual77_seed, 0u, pert_fcbias_fp32_mode2__dual77_local_size, pert_fcbias_fp32_mode2__dual77_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref;
} _pert_fcbias_fp32_mode2__dual77_cluster_fork_args_t;

static void _pert_fcbias_fp32_mode2__dual77_cluster_fork(void *_pert_fcbias_fp32_mode2__dual77_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual77_cluster_fork_args_t *args =
      (_pert_fcbias_fp32_mode2__dual77_cluster_fork_args_t *)_pert_fcbias_fp32_mode2__dual77_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode2__dual77_tiling_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual77_tiling_closure_args =
      (_pert_fcbias_fp32_mode2__dual77_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref};

  // _pert_fcbias_fp32_mode2__dual77_tiling_closure CLOSURE CALL
  _pert_fcbias_fp32_mode2__dual77_tiling_closure(&DeeployNetwork__pert_fcbias_fp32_mode2__dual77_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr;
} _pert_fcbias_fp32_mode2__dual77_closure_args_t;

static void _pert_fcbias_fp32_mode2__dual77_closure(void *_pert_fcbias_fp32_mode2__dual77_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual77_closure_args_t *args = (_pert_fcbias_fp32_mode2__dual77_closure_args_t *)_pert_fcbias_fp32_mode2__dual77_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_input_23_ref =
      (void *)((char *)DeeployNetwork_input_23 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_fcbias_fp32_pert_m__dual76_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_input_23_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcbias_fp32_mode2__dual77_cluster_fork_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual77_cluster_fork_args =
        (_pert_fcbias_fp32_mode2__dual77_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcbias_fp32_mode2__dual77_cluster_fork, &DeeployNetwork__pert_fcbias_fp32_mode2__dual77_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_fcbias_fp32_pert_m__dual76_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr;
} _pert_fcbias_fp32_mode2__dual77_closure_L3_args_t;

static void _pert_fcbias_fp32_mode2__dual77_closure_L3(void *_pert_fcbias_fp32_mode2__dual77_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual77_closure_L3_args_t *args =
      (_pert_fcbias_fp32_mode2__dual77_closure_L3_args_t *)_pert_fcbias_fp32_mode2__dual77_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode2__dual77_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual77_closure_args = (_pert_fcbias_fp32_mode2__dual77_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr};

  // _pert_fcbias_fp32_mode2__dual77_closure CLOSURE CALL
  _pert_fcbias_fp32_mode2__dual77_closure(&DeeployNetwork__pert_fcbias_fp32_mode2__dual77_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref;
} _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args_t;

static void _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure(void *_wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args_t *)_wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: wrappedInnerForwardImplConv_ne16bias__dual4, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t wrappedInnerForwardImplConv_ne16bias__dual4_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
    uint32_t wrappedInnerForwardImplConv_ne16bias__dual4_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 1);
    uint32_t wrappedInnerForwardImplConv_ne16bias__dual4_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImplConv_ne16bias__dual4_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref,
                      (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref,
                      (int8_t *)DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer, wrappedInnerForwardImplConv_ne16bias__dual4_seed_w,
                      wrappedInnerForwardImplConv_ne16bias__dual4_seed_b, wrappedInnerForwardImplConv_ne16bias__dual4_tile_xor, 15, 16,
                      wrappedInnerForwardImplConv_ne16bias__dual4_eps_scale, NUM_CORES, 8, 4, 128, 0,
                      (uint32_t *)DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref;
} _wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args_t;

static void _wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork(void *_wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args_t *)_wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer = DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref};

  // _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure(&DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr;
} _wrappedInnerForwardImplConv_ne16bias__dual4_closure_args_t;

static void _wrappedInnerForwardImplConv_ne16bias__dual4_closure(void *_wrappedInnerForwardImplConv_ne16bias__dual4_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16bias__dual4_closure_args_t *args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_closure_args_t *)_wrappedInnerForwardImplConv_ne16bias__dual4_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 288);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 192);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_input_12_ref =
      (void *)((char *)DeeployNetwork_input_12 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16bias__dual4_ref =
          (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16bias__dual4 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_input_17_ref =
      (void *)((char *)DeeployNetwork_input_17 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convbias_rqsadd_ne16add2__dual2_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_input_12_ref);
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16bias__dual4_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_input_17_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convbias_rqsadd_pmul_tensor_ref);
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args =
        (_wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer = DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_bmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_blocks0convbias_rqsadd_ne16add2__dual2_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr;
} _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args_t;

static void _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3(void *_wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args_t *)_wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_ne16bias__dual4_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_closure_args =
      (_wrappedInnerForwardImplConv_ne16bias__dual4_closure_args_t){
          .DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr};

  // _wrappedInnerForwardImplConv_ne16bias__dual4_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_ne16bias__dual4_closure(&DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref;
} _pert_blocks0bnweight_mode0__dual14_tiling_closure_args_t;

static void _pert_blocks0bnweight_mode0__dual14_tiling_closure(void *_pert_blocks0bnweight_mode0__dual14_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual14_tiling_closure_args_t *args =
      (_pert_blocks0bnweight_mode0__dual14_tiling_closure_args_t *)_pert_blocks0bnweight_mode0__dual14_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks0bnweight_mode0__dual14, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks0bnweight_mode0__dual14_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnweight_mode0__dual14_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnweight_mode0__dual14_chunk = (8 >> pert_blocks0bnweight_mode0__dual14_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnweight_mode0__dual14_chunk_start =
      (uint32_t)MIN(pert_blocks0bnweight_mode0__dual14_chunk * pert_blocks0bnweight_mode0__dual14_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnweight_mode0__dual14_chunk_stop =
      (uint32_t)MIN(pert_blocks0bnweight_mode0__dual14_chunk_start + pert_blocks0bnweight_mode0__dual14_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnweight_mode0__dual14_local_size = pert_blocks0bnweight_mode0__dual14_chunk_stop - pert_blocks0bnweight_mode0__dual14_chunk_start;
  uint32_t pert_blocks0bnweight_mode0__dual14_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 12 + pert_blocks0bnweight_mode0__dual14_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnweight_mode0__dual14_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref[pert_blocks0bnweight_mode0__dual14_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref)[pert_blocks0bnweight_mode0__dual14_chunk_start],
      pert_blocks0bnweight_mode0__dual14_seed, 1u, pert_blocks0bnweight_mode0__dual14_local_size, pert_blocks0bnweight_mode0__dual14_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref[pert_blocks0bnweight_mode0__dual14_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref[8 + pert_blocks0bnweight_mode0__dual14_chunk_start],
      pert_blocks0bnweight_mode0__dual14_seed, 0u, pert_blocks0bnweight_mode0__dual14_local_size, pert_blocks0bnweight_mode0__dual14_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref;
} _pert_blocks0bnweight_mode0__dual14_cluster_fork_args_t;

static void _pert_blocks0bnweight_mode0__dual14_cluster_fork(void *_pert_blocks0bnweight_mode0__dual14_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual14_cluster_fork_args_t *args =
      (_pert_blocks0bnweight_mode0__dual14_cluster_fork_args_t *)_pert_blocks0bnweight_mode0__dual14_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_mode0__dual14_tiling_closure_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual14_tiling_closure_args =
      (_pert_blocks0bnweight_mode0__dual14_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref};

  // _pert_blocks0bnweight_mode0__dual14_tiling_closure CLOSURE CALL
  _pert_blocks0bnweight_mode0__dual14_tiling_closure(&DeeployNetwork__pert_blocks0bnweight_mode0__dual14_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr;
} _pert_blocks0bnweight_mode0__dual14_closure_args_t;

static void _pert_blocks0bnweight_mode0__dual14_closure(void *_pert_blocks0bnweight_mode0__dual14_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual14_closure_args_t *args =
      (_pert_blocks0bnweight_mode0__dual14_closure_args_t *)_pert_blocks0bnweight_mode0__dual14_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_blocks0bnweight_pert_pm__dual13_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_input_2_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks0bnweight_mode0__dual14_cluster_fork_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual14_cluster_fork_args =
        (_pert_blocks0bnweight_mode0__dual14_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnweight_mode0__dual14_cluster_fork, &DeeployNetwork__pert_blocks0bnweight_mode0__dual14_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_blocks0bnweight_pert_pm__dual13_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr;
} _pert_blocks0bnweight_mode0__dual14_closure_L3_args_t;

static void _pert_blocks0bnweight_mode0__dual14_closure_L3(void *_pert_blocks0bnweight_mode0__dual14_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual14_closure_L3_args_t *args =
      (_pert_blocks0bnweight_mode0__dual14_closure_L3_args_t *)_pert_blocks0bnweight_mode0__dual14_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_mode0__dual14_closure_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual14_closure_args =
      (_pert_blocks0bnweight_mode0__dual14_closure_args_t){.DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor =
                                                               DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr};

  // _pert_blocks0bnweight_mode0__dual14_closure CLOSURE CALL
  _pert_blocks0bnweight_mode0__dual14_closure(&DeeployNetwork__pert_blocks0bnweight_mode0__dual14_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref;
} _pert_blocks0bnbias_mode0__dual16_tiling_closure_args_t;

static void _pert_blocks0bnbias_mode0__dual16_tiling_closure(void *_pert_blocks0bnbias_mode0__dual16_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual16_tiling_closure_args_t *args =
      (_pert_blocks0bnbias_mode0__dual16_tiling_closure_args_t *)_pert_blocks0bnbias_mode0__dual16_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks0bnbias_mode0__dual16, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks0bnbias_mode0__dual16_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnbias_mode0__dual16_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnbias_mode0__dual16_chunk = (8 >> pert_blocks0bnbias_mode0__dual16_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnbias_mode0__dual16_chunk_start =
      (uint32_t)MIN(pert_blocks0bnbias_mode0__dual16_chunk * pert_blocks0bnbias_mode0__dual16_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnbias_mode0__dual16_chunk_stop =
      (uint32_t)MIN(pert_blocks0bnbias_mode0__dual16_chunk_start + pert_blocks0bnbias_mode0__dual16_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnbias_mode0__dual16_local_size = pert_blocks0bnbias_mode0__dual16_chunk_stop - pert_blocks0bnbias_mode0__dual16_chunk_start;
  uint32_t pert_blocks0bnbias_mode0__dual16_seed = ((42 + perturb_seed_base) + NUM_CORES * 13 + pert_blocks0bnbias_mode0__dual16_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnbias_mode0__dual16_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref[pert_blocks0bnbias_mode0__dual16_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref)[pert_blocks0bnbias_mode0__dual16_chunk_start],
      pert_blocks0bnbias_mode0__dual16_seed, 1u, pert_blocks0bnbias_mode0__dual16_local_size, pert_blocks0bnbias_mode0__dual16_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref[pert_blocks0bnbias_mode0__dual16_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref[8 + pert_blocks0bnbias_mode0__dual16_chunk_start],
      pert_blocks0bnbias_mode0__dual16_seed, 0u, pert_blocks0bnbias_mode0__dual16_local_size, pert_blocks0bnbias_mode0__dual16_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref;
} _pert_blocks0bnbias_mode0__dual16_cluster_fork_args_t;

static void _pert_blocks0bnbias_mode0__dual16_cluster_fork(void *_pert_blocks0bnbias_mode0__dual16_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual16_cluster_fork_args_t *args =
      (_pert_blocks0bnbias_mode0__dual16_cluster_fork_args_t *)_pert_blocks0bnbias_mode0__dual16_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_mode0__dual16_tiling_closure_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual16_tiling_closure_args =
      (_pert_blocks0bnbias_mode0__dual16_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref};

  // _pert_blocks0bnbias_mode0__dual16_tiling_closure CLOSURE CALL
  _pert_blocks0bnbias_mode0__dual16_tiling_closure(&DeeployNetwork__pert_blocks0bnbias_mode0__dual16_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr;
} _pert_blocks0bnbias_mode0__dual16_closure_args_t;

static void _pert_blocks0bnbias_mode0__dual16_closure(void *_pert_blocks0bnbias_mode0__dual16_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual16_closure_args_t *args = (_pert_blocks0bnbias_mode0__dual16_closure_args_t *)_pert_blocks0bnbias_mode0__dual16_closure_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_blocks0bnbias_pert_pm__dual15_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks0bnbias_mode0__dual16_cluster_fork_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual16_cluster_fork_args =
        (_pert_blocks0bnbias_mode0__dual16_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnbias_mode0__dual16_cluster_fork, &DeeployNetwork__pert_blocks0bnbias_mode0__dual16_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_blocks0bnbias_pert_pm__dual15_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr;
} _pert_blocks0bnbias_mode0__dual16_closure_L3_args_t;

static void _pert_blocks0bnbias_mode0__dual16_closure_L3(void *_pert_blocks0bnbias_mode0__dual16_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual16_closure_L3_args_t *args =
      (_pert_blocks0bnbias_mode0__dual16_closure_L3_args_t *)_pert_blocks0bnbias_mode0__dual16_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_mode0__dual16_closure_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual16_closure_args =
      (_pert_blocks0bnbias_mode0__dual16_closure_args_t){.DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor =
                                                             DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr};

  // _pert_blocks0bnbias_mode0__dual16_closure CLOSURE CALL
  _pert_blocks0bnbias_mode0__dual16_closure(&DeeployNetwork__pert_blocks0bnbias_mode0__dual16_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure(void *_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: wrappedInnerForwardImpl_1Conv_ne16bias__dual22, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t wrappedInnerForwardImpl_1Conv_ne16bias__dual22_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
    uint32_t wrappedInnerForwardImpl_1Conv_ne16bias__dual22_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 3);
    uint32_t wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImpl_1Conv_ne16bias__dual22_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref,
                      (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref,
                      (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer, wrappedInnerForwardImpl_1Conv_ne16bias__dual22_seed_w,
                      wrappedInnerForwardImpl_1Conv_ne16bias__dual22_seed_b, wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tile_xor, 15, 16,
                      wrappedInnerForwardImpl_1Conv_ne16bias__dual22_eps_scale, NUM_CORES, 16, 128, 0, 0,
                      (uint32_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork(void *_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args_t *)_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure(void *_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6464);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6528);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref =
      (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6336);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_input_13_ref =
      (void *)((char *)DeeployNetwork_input_13 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_ref =
          (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_input_18_ref =
      (void *)((char *)DeeployNetwork_input_18 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_ref =
      (void
           *)((char *)
                  DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22 +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convbias_rqsadd_ne16add2__dual20_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2112);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_input_13_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_input_18_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convbias_rqsadd_pmul_tensor_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args =
        (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_bmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_blocks1convbias_rqsadd_ne16add2__dual20_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3(void *_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args_t *)_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args =
      (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args_t){
          .DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref;
} _pert_blocks1bnweight_mode0__dual29_tiling_closure_args_t;

static void _pert_blocks1bnweight_mode0__dual29_tiling_closure(void *_pert_blocks1bnweight_mode0__dual29_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual29_tiling_closure_args_t *args =
      (_pert_blocks1bnweight_mode0__dual29_tiling_closure_args_t *)_pert_blocks1bnweight_mode0__dual29_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks1bnweight_mode0__dual29, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks1bnweight_mode0__dual29_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnweight_mode0__dual29_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnweight_mode0__dual29_chunk = (16 >> pert_blocks1bnweight_mode0__dual29_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnweight_mode0__dual29_chunk_start =
      (uint32_t)MIN(pert_blocks1bnweight_mode0__dual29_chunk * pert_blocks1bnweight_mode0__dual29_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnweight_mode0__dual29_chunk_stop =
      (uint32_t)MIN(pert_blocks1bnweight_mode0__dual29_chunk_start + pert_blocks1bnweight_mode0__dual29_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnweight_mode0__dual29_local_size = pert_blocks1bnweight_mode0__dual29_chunk_stop - pert_blocks1bnweight_mode0__dual29_chunk_start;
  uint32_t pert_blocks1bnweight_mode0__dual29_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 14 + pert_blocks1bnweight_mode0__dual29_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnweight_mode0__dual29_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref[pert_blocks1bnweight_mode0__dual29_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref)[pert_blocks1bnweight_mode0__dual29_chunk_start],
      pert_blocks1bnweight_mode0__dual29_seed, 1u, pert_blocks1bnweight_mode0__dual29_local_size, pert_blocks1bnweight_mode0__dual29_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref[pert_blocks1bnweight_mode0__dual29_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref[16 + pert_blocks1bnweight_mode0__dual29_chunk_start],
      pert_blocks1bnweight_mode0__dual29_seed, 0u, pert_blocks1bnweight_mode0__dual29_local_size, pert_blocks1bnweight_mode0__dual29_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref;
} _pert_blocks1bnweight_mode0__dual29_cluster_fork_args_t;

static void _pert_blocks1bnweight_mode0__dual29_cluster_fork(void *_pert_blocks1bnweight_mode0__dual29_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual29_cluster_fork_args_t *args =
      (_pert_blocks1bnweight_mode0__dual29_cluster_fork_args_t *)_pert_blocks1bnweight_mode0__dual29_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_mode0__dual29_tiling_closure_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual29_tiling_closure_args =
      (_pert_blocks1bnweight_mode0__dual29_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref};

  // _pert_blocks1bnweight_mode0__dual29_tiling_closure CLOSURE CALL
  _pert_blocks1bnweight_mode0__dual29_tiling_closure(&DeeployNetwork__pert_blocks1bnweight_mode0__dual29_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr;
} _pert_blocks1bnweight_mode0__dual29_closure_args_t;

static void _pert_blocks1bnweight_mode0__dual29_closure(void *_pert_blocks1bnweight_mode0__dual29_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual29_closure_args_t *args =
      (_pert_blocks1bnweight_mode0__dual29_closure_args_t *)_pert_blocks1bnweight_mode0__dual29_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_input_4_ref =
      (void *)((char *)DeeployNetwork_input_4 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_blocks1bnweight_pert_pm__dual28_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_input_4_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks1bnweight_mode0__dual29_cluster_fork_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual29_cluster_fork_args =
        (_pert_blocks1bnweight_mode0__dual29_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnweight_mode0__dual29_cluster_fork, &DeeployNetwork__pert_blocks1bnweight_mode0__dual29_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_blocks1bnweight_pert_pm__dual28_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr;
} _pert_blocks1bnweight_mode0__dual29_closure_L3_args_t;

static void _pert_blocks1bnweight_mode0__dual29_closure_L3(void *_pert_blocks1bnweight_mode0__dual29_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual29_closure_L3_args_t *args =
      (_pert_blocks1bnweight_mode0__dual29_closure_L3_args_t *)_pert_blocks1bnweight_mode0__dual29_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_mode0__dual29_closure_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual29_closure_args =
      (_pert_blocks1bnweight_mode0__dual29_closure_args_t){.DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor =
                                                               DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr};

  // _pert_blocks1bnweight_mode0__dual29_closure CLOSURE CALL
  _pert_blocks1bnweight_mode0__dual29_closure(&DeeployNetwork__pert_blocks1bnweight_mode0__dual29_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref;
} _pert_blocks1bnbias_mode0__dual31_tiling_closure_args_t;

static void _pert_blocks1bnbias_mode0__dual31_tiling_closure(void *_pert_blocks1bnbias_mode0__dual31_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual31_tiling_closure_args_t *args =
      (_pert_blocks1bnbias_mode0__dual31_tiling_closure_args_t *)_pert_blocks1bnbias_mode0__dual31_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks1bnbias_mode0__dual31, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks1bnbias_mode0__dual31_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnbias_mode0__dual31_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnbias_mode0__dual31_chunk = (16 >> pert_blocks1bnbias_mode0__dual31_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnbias_mode0__dual31_chunk_start =
      (uint32_t)MIN(pert_blocks1bnbias_mode0__dual31_chunk * pert_blocks1bnbias_mode0__dual31_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnbias_mode0__dual31_chunk_stop =
      (uint32_t)MIN(pert_blocks1bnbias_mode0__dual31_chunk_start + pert_blocks1bnbias_mode0__dual31_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnbias_mode0__dual31_local_size = pert_blocks1bnbias_mode0__dual31_chunk_stop - pert_blocks1bnbias_mode0__dual31_chunk_start;
  uint32_t pert_blocks1bnbias_mode0__dual31_seed = ((42 + perturb_seed_base) + NUM_CORES * 15 + pert_blocks1bnbias_mode0__dual31_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnbias_mode0__dual31_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref[pert_blocks1bnbias_mode0__dual31_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref)[pert_blocks1bnbias_mode0__dual31_chunk_start],
      pert_blocks1bnbias_mode0__dual31_seed, 1u, pert_blocks1bnbias_mode0__dual31_local_size, pert_blocks1bnbias_mode0__dual31_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref[pert_blocks1bnbias_mode0__dual31_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref[16 + pert_blocks1bnbias_mode0__dual31_chunk_start],
      pert_blocks1bnbias_mode0__dual31_seed, 0u, pert_blocks1bnbias_mode0__dual31_local_size, pert_blocks1bnbias_mode0__dual31_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref;
} _pert_blocks1bnbias_mode0__dual31_cluster_fork_args_t;

static void _pert_blocks1bnbias_mode0__dual31_cluster_fork(void *_pert_blocks1bnbias_mode0__dual31_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual31_cluster_fork_args_t *args =
      (_pert_blocks1bnbias_mode0__dual31_cluster_fork_args_t *)_pert_blocks1bnbias_mode0__dual31_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_mode0__dual31_tiling_closure_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual31_tiling_closure_args =
      (_pert_blocks1bnbias_mode0__dual31_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref};

  // _pert_blocks1bnbias_mode0__dual31_tiling_closure CLOSURE CALL
  _pert_blocks1bnbias_mode0__dual31_tiling_closure(&DeeployNetwork__pert_blocks1bnbias_mode0__dual31_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr;
} _pert_blocks1bnbias_mode0__dual31_closure_args_t;

static void _pert_blocks1bnbias_mode0__dual31_closure(void *_pert_blocks1bnbias_mode0__dual31_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual31_closure_args_t *args = (_pert_blocks1bnbias_mode0__dual31_closure_args_t *)_pert_blocks1bnbias_mode0__dual31_closure_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_input_5_ref =
      (void *)((char *)DeeployNetwork_input_5 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_blocks1bnbias_pert_pm__dual30_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_input_5_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks1bnbias_mode0__dual31_cluster_fork_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual31_cluster_fork_args =
        (_pert_blocks1bnbias_mode0__dual31_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnbias_mode0__dual31_cluster_fork, &DeeployNetwork__pert_blocks1bnbias_mode0__dual31_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_blocks1bnbias_pert_pm__dual30_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr;
} _pert_blocks1bnbias_mode0__dual31_closure_L3_args_t;

static void _pert_blocks1bnbias_mode0__dual31_closure_L3(void *_pert_blocks1bnbias_mode0__dual31_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual31_closure_L3_args_t *args =
      (_pert_blocks1bnbias_mode0__dual31_closure_L3_args_t *)_pert_blocks1bnbias_mode0__dual31_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_mode0__dual31_closure_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual31_closure_args =
      (_pert_blocks1bnbias_mode0__dual31_closure_args_t){.DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor =
                                                             DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr};

  // _pert_blocks1bnbias_mode0__dual31_closure CLOSURE CALL
  _pert_blocks1bnbias_mode0__dual31_closure(&DeeployNetwork__pert_blocks1bnbias_mode0__dual31_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref;
} _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure(void *_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: wrappedInnerForwardImpl_2Conv_ne16bias__dual37, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t wrappedInnerForwardImpl_2Conv_ne16bias__dual37_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 4);
    uint32_t wrappedInnerForwardImpl_2Conv_ne16bias__dual37_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 5);
    uint32_t wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImpl_2Conv_ne16bias__dual37_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref,
                      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref,
                      (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref,
                      (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer, wrappedInnerForwardImpl_2Conv_ne16bias__dual37_seed_w,
                      wrappedInnerForwardImpl_2Conv_ne16bias__dual37_seed_b, wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tile_xor, 15, 16,
                      wrappedInnerForwardImpl_2Conv_ne16bias__dual37_eps_scale, NUM_CORES, 16, 128, 0, 0,
                      (uint32_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref;
} _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork(void *_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args_t *)_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer = DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref};

  // _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure(void *_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6464);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6528);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref =
      (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6336);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_input_14_ref =
      (void *)((char *)DeeployNetwork_input_14 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_ref =
          (void *)((char *)DeeployNetwork_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_input_19_ref =
      (void *)((char *)DeeployNetwork_input_19 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_ref =
      (void
           *)((char *)
                  DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37 +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convbias_rqsadd_ne16add2__dual35_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2112);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_input_14_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_input_19_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convbias_rqsadd_pmul_tensor_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args =
        (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer = DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_bmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_blocks2convbias_rqsadd_ne16add2__dual35_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3(void *_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args_t *)_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args =
      (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args_t){
          .DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr};

  // _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref;
} _pert_blocks2bnweight_mode0__dual44_tiling_closure_args_t;

static void _pert_blocks2bnweight_mode0__dual44_tiling_closure(void *_pert_blocks2bnweight_mode0__dual44_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual44_tiling_closure_args_t *args =
      (_pert_blocks2bnweight_mode0__dual44_tiling_closure_args_t *)_pert_blocks2bnweight_mode0__dual44_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks2bnweight_mode0__dual44, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks2bnweight_mode0__dual44_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnweight_mode0__dual44_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnweight_mode0__dual44_chunk = (16 >> pert_blocks2bnweight_mode0__dual44_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnweight_mode0__dual44_chunk_start =
      (uint32_t)MIN(pert_blocks2bnweight_mode0__dual44_chunk * pert_blocks2bnweight_mode0__dual44_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnweight_mode0__dual44_chunk_stop =
      (uint32_t)MIN(pert_blocks2bnweight_mode0__dual44_chunk_start + pert_blocks2bnweight_mode0__dual44_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnweight_mode0__dual44_local_size = pert_blocks2bnweight_mode0__dual44_chunk_stop - pert_blocks2bnweight_mode0__dual44_chunk_start;
  uint32_t pert_blocks2bnweight_mode0__dual44_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 16 + pert_blocks2bnweight_mode0__dual44_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnweight_mode0__dual44_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref[pert_blocks2bnweight_mode0__dual44_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref)[pert_blocks2bnweight_mode0__dual44_chunk_start],
      pert_blocks2bnweight_mode0__dual44_seed, 1u, pert_blocks2bnweight_mode0__dual44_local_size, pert_blocks2bnweight_mode0__dual44_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref[pert_blocks2bnweight_mode0__dual44_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref[16 + pert_blocks2bnweight_mode0__dual44_chunk_start],
      pert_blocks2bnweight_mode0__dual44_seed, 0u, pert_blocks2bnweight_mode0__dual44_local_size, pert_blocks2bnweight_mode0__dual44_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref;
} _pert_blocks2bnweight_mode0__dual44_cluster_fork_args_t;

static void _pert_blocks2bnweight_mode0__dual44_cluster_fork(void *_pert_blocks2bnweight_mode0__dual44_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual44_cluster_fork_args_t *args =
      (_pert_blocks2bnweight_mode0__dual44_cluster_fork_args_t *)_pert_blocks2bnweight_mode0__dual44_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_mode0__dual44_tiling_closure_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual44_tiling_closure_args =
      (_pert_blocks2bnweight_mode0__dual44_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref};

  // _pert_blocks2bnweight_mode0__dual44_tiling_closure CLOSURE CALL
  _pert_blocks2bnweight_mode0__dual44_tiling_closure(&DeeployNetwork__pert_blocks2bnweight_mode0__dual44_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr;
} _pert_blocks2bnweight_mode0__dual44_closure_args_t;

static void _pert_blocks2bnweight_mode0__dual44_closure(void *_pert_blocks2bnweight_mode0__dual44_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual44_closure_args_t *args =
      (_pert_blocks2bnweight_mode0__dual44_closure_args_t *)_pert_blocks2bnweight_mode0__dual44_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_input_6_ref =
      (void *)((char *)DeeployNetwork_input_6 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_blocks2bnweight_pert_pm__dual43_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_input_6_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks2bnweight_mode0__dual44_cluster_fork_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual44_cluster_fork_args =
        (_pert_blocks2bnweight_mode0__dual44_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnweight_mode0__dual44_cluster_fork, &DeeployNetwork__pert_blocks2bnweight_mode0__dual44_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_blocks2bnweight_pert_pm__dual43_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr;
} _pert_blocks2bnweight_mode0__dual44_closure_L3_args_t;

static void _pert_blocks2bnweight_mode0__dual44_closure_L3(void *_pert_blocks2bnweight_mode0__dual44_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual44_closure_L3_args_t *args =
      (_pert_blocks2bnweight_mode0__dual44_closure_L3_args_t *)_pert_blocks2bnweight_mode0__dual44_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_mode0__dual44_closure_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual44_closure_args =
      (_pert_blocks2bnweight_mode0__dual44_closure_args_t){.DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor =
                                                               DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr};

  // _pert_blocks2bnweight_mode0__dual44_closure CLOSURE CALL
  _pert_blocks2bnweight_mode0__dual44_closure(&DeeployNetwork__pert_blocks2bnweight_mode0__dual44_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref;
} _pert_blocks2bnbias_mode0__dual46_tiling_closure_args_t;

static void _pert_blocks2bnbias_mode0__dual46_tiling_closure(void *_pert_blocks2bnbias_mode0__dual46_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual46_tiling_closure_args_t *args =
      (_pert_blocks2bnbias_mode0__dual46_tiling_closure_args_t *)_pert_blocks2bnbias_mode0__dual46_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks2bnbias_mode0__dual46, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks2bnbias_mode0__dual46_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnbias_mode0__dual46_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnbias_mode0__dual46_chunk = (16 >> pert_blocks2bnbias_mode0__dual46_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnbias_mode0__dual46_chunk_start =
      (uint32_t)MIN(pert_blocks2bnbias_mode0__dual46_chunk * pert_blocks2bnbias_mode0__dual46_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnbias_mode0__dual46_chunk_stop =
      (uint32_t)MIN(pert_blocks2bnbias_mode0__dual46_chunk_start + pert_blocks2bnbias_mode0__dual46_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnbias_mode0__dual46_local_size = pert_blocks2bnbias_mode0__dual46_chunk_stop - pert_blocks2bnbias_mode0__dual46_chunk_start;
  uint32_t pert_blocks2bnbias_mode0__dual46_seed = ((42 + perturb_seed_base) + NUM_CORES * 17 + pert_blocks2bnbias_mode0__dual46_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnbias_mode0__dual46_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref[pert_blocks2bnbias_mode0__dual46_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref)[pert_blocks2bnbias_mode0__dual46_chunk_start],
      pert_blocks2bnbias_mode0__dual46_seed, 1u, pert_blocks2bnbias_mode0__dual46_local_size, pert_blocks2bnbias_mode0__dual46_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref[pert_blocks2bnbias_mode0__dual46_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref[16 + pert_blocks2bnbias_mode0__dual46_chunk_start],
      pert_blocks2bnbias_mode0__dual46_seed, 0u, pert_blocks2bnbias_mode0__dual46_local_size, pert_blocks2bnbias_mode0__dual46_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref;
} _pert_blocks2bnbias_mode0__dual46_cluster_fork_args_t;

static void _pert_blocks2bnbias_mode0__dual46_cluster_fork(void *_pert_blocks2bnbias_mode0__dual46_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual46_cluster_fork_args_t *args =
      (_pert_blocks2bnbias_mode0__dual46_cluster_fork_args_t *)_pert_blocks2bnbias_mode0__dual46_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_mode0__dual46_tiling_closure_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual46_tiling_closure_args =
      (_pert_blocks2bnbias_mode0__dual46_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref};

  // _pert_blocks2bnbias_mode0__dual46_tiling_closure CLOSURE CALL
  _pert_blocks2bnbias_mode0__dual46_tiling_closure(&DeeployNetwork__pert_blocks2bnbias_mode0__dual46_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr;
} _pert_blocks2bnbias_mode0__dual46_closure_args_t;

static void _pert_blocks2bnbias_mode0__dual46_closure(void *_pert_blocks2bnbias_mode0__dual46_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual46_closure_args_t *args = (_pert_blocks2bnbias_mode0__dual46_closure_args_t *)_pert_blocks2bnbias_mode0__dual46_closure_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_input_7_ref =
      (void *)((char *)DeeployNetwork_input_7 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_blocks2bnbias_pert_pm__dual45_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_input_7_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks2bnbias_mode0__dual46_cluster_fork_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual46_cluster_fork_args =
        (_pert_blocks2bnbias_mode0__dual46_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnbias_mode0__dual46_cluster_fork, &DeeployNetwork__pert_blocks2bnbias_mode0__dual46_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_blocks2bnbias_pert_pm__dual45_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr;
} _pert_blocks2bnbias_mode0__dual46_closure_L3_args_t;

static void _pert_blocks2bnbias_mode0__dual46_closure_L3(void *_pert_blocks2bnbias_mode0__dual46_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual46_closure_L3_args_t *args =
      (_pert_blocks2bnbias_mode0__dual46_closure_L3_args_t *)_pert_blocks2bnbias_mode0__dual46_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_mode0__dual46_closure_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual46_closure_args =
      (_pert_blocks2bnbias_mode0__dual46_closure_args_t){.DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor =
                                                             DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr};

  // _pert_blocks2bnbias_mode0__dual46_closure CLOSURE CALL
  _pert_blocks2bnbias_mode0__dual46_closure(&DeeployNetwork__pert_blocks2bnbias_mode0__dual46_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref;
} _pert_blocks3bnweight_mode0__dual51_tiling_closure_args_t;

static void _pert_blocks3bnweight_mode0__dual51_tiling_closure(void *_pert_blocks3bnweight_mode0__dual51_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual51_tiling_closure_args_t *args =
      (_pert_blocks3bnweight_mode0__dual51_tiling_closure_args_t *)_pert_blocks3bnweight_mode0__dual51_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks3bnweight_mode0__dual51, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks3bnweight_mode0__dual51_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnweight_mode0__dual51_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnweight_mode0__dual51_chunk = (32 >> pert_blocks3bnweight_mode0__dual51_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnweight_mode0__dual51_chunk_start =
      (uint32_t)MIN(pert_blocks3bnweight_mode0__dual51_chunk * pert_blocks3bnweight_mode0__dual51_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnweight_mode0__dual51_chunk_stop =
      (uint32_t)MIN(pert_blocks3bnweight_mode0__dual51_chunk_start + pert_blocks3bnweight_mode0__dual51_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnweight_mode0__dual51_local_size = pert_blocks3bnweight_mode0__dual51_chunk_stop - pert_blocks3bnweight_mode0__dual51_chunk_start;
  uint32_t pert_blocks3bnweight_mode0__dual51_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 18 + pert_blocks3bnweight_mode0__dual51_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnweight_mode0__dual51_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref[pert_blocks3bnweight_mode0__dual51_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref)[pert_blocks3bnweight_mode0__dual51_chunk_start],
      pert_blocks3bnweight_mode0__dual51_seed, 1u, pert_blocks3bnweight_mode0__dual51_local_size, pert_blocks3bnweight_mode0__dual51_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref[pert_blocks3bnweight_mode0__dual51_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref[32 + pert_blocks3bnweight_mode0__dual51_chunk_start],
      pert_blocks3bnweight_mode0__dual51_seed, 0u, pert_blocks3bnweight_mode0__dual51_local_size, pert_blocks3bnweight_mode0__dual51_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref;
} _pert_blocks3bnweight_mode0__dual51_cluster_fork_args_t;

static void _pert_blocks3bnweight_mode0__dual51_cluster_fork(void *_pert_blocks3bnweight_mode0__dual51_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual51_cluster_fork_args_t *args =
      (_pert_blocks3bnweight_mode0__dual51_cluster_fork_args_t *)_pert_blocks3bnweight_mode0__dual51_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_mode0__dual51_tiling_closure_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual51_tiling_closure_args =
      (_pert_blocks3bnweight_mode0__dual51_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref};

  // _pert_blocks3bnweight_mode0__dual51_tiling_closure CLOSURE CALL
  _pert_blocks3bnweight_mode0__dual51_tiling_closure(&DeeployNetwork__pert_blocks3bnweight_mode0__dual51_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr;
} _pert_blocks3bnweight_mode0__dual51_closure_args_t;

static void _pert_blocks3bnweight_mode0__dual51_closure(void *_pert_blocks3bnweight_mode0__dual51_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual51_closure_args_t *args =
      (_pert_blocks3bnweight_mode0__dual51_closure_args_t *)_pert_blocks3bnweight_mode0__dual51_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_input_8_ref =
      (void *)((char *)DeeployNetwork_input_8 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_blocks3bnweight_pert_pm__dual50_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_input_8_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks3bnweight_mode0__dual51_cluster_fork_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual51_cluster_fork_args =
        (_pert_blocks3bnweight_mode0__dual51_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnweight_mode0__dual51_cluster_fork, &DeeployNetwork__pert_blocks3bnweight_mode0__dual51_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_blocks3bnweight_pert_pm__dual50_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr;
} _pert_blocks3bnweight_mode0__dual51_closure_L3_args_t;

static void _pert_blocks3bnweight_mode0__dual51_closure_L3(void *_pert_blocks3bnweight_mode0__dual51_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual51_closure_L3_args_t *args =
      (_pert_blocks3bnweight_mode0__dual51_closure_L3_args_t *)_pert_blocks3bnweight_mode0__dual51_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_mode0__dual51_closure_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual51_closure_args =
      (_pert_blocks3bnweight_mode0__dual51_closure_args_t){.DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor =
                                                               DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr};

  // _pert_blocks3bnweight_mode0__dual51_closure CLOSURE CALL
  _pert_blocks3bnweight_mode0__dual51_closure(&DeeployNetwork__pert_blocks3bnweight_mode0__dual51_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref;
} _pert_blocks3bnbias_mode0__dual53_tiling_closure_args_t;

static void _pert_blocks3bnbias_mode0__dual53_tiling_closure(void *_pert_blocks3bnbias_mode0__dual53_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual53_tiling_closure_args_t *args =
      (_pert_blocks3bnbias_mode0__dual53_tiling_closure_args_t *)_pert_blocks3bnbias_mode0__dual53_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks3bnbias_mode0__dual53, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks3bnbias_mode0__dual53_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnbias_mode0__dual53_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnbias_mode0__dual53_chunk = (32 >> pert_blocks3bnbias_mode0__dual53_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnbias_mode0__dual53_chunk_start =
      (uint32_t)MIN(pert_blocks3bnbias_mode0__dual53_chunk * pert_blocks3bnbias_mode0__dual53_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnbias_mode0__dual53_chunk_stop =
      (uint32_t)MIN(pert_blocks3bnbias_mode0__dual53_chunk_start + pert_blocks3bnbias_mode0__dual53_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnbias_mode0__dual53_local_size = pert_blocks3bnbias_mode0__dual53_chunk_stop - pert_blocks3bnbias_mode0__dual53_chunk_start;
  uint32_t pert_blocks3bnbias_mode0__dual53_seed = ((42 + perturb_seed_base) + NUM_CORES * 19 + pert_blocks3bnbias_mode0__dual53_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnbias_mode0__dual53_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref[pert_blocks3bnbias_mode0__dual53_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref)[pert_blocks3bnbias_mode0__dual53_chunk_start],
      pert_blocks3bnbias_mode0__dual53_seed, 1u, pert_blocks3bnbias_mode0__dual53_local_size, pert_blocks3bnbias_mode0__dual53_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref[pert_blocks3bnbias_mode0__dual53_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref[32 + pert_blocks3bnbias_mode0__dual53_chunk_start],
      pert_blocks3bnbias_mode0__dual53_seed, 0u, pert_blocks3bnbias_mode0__dual53_local_size, pert_blocks3bnbias_mode0__dual53_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref;
} _pert_blocks3bnbias_mode0__dual53_cluster_fork_args_t;

static void _pert_blocks3bnbias_mode0__dual53_cluster_fork(void *_pert_blocks3bnbias_mode0__dual53_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual53_cluster_fork_args_t *args =
      (_pert_blocks3bnbias_mode0__dual53_cluster_fork_args_t *)_pert_blocks3bnbias_mode0__dual53_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_mode0__dual53_tiling_closure_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual53_tiling_closure_args =
      (_pert_blocks3bnbias_mode0__dual53_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref};

  // _pert_blocks3bnbias_mode0__dual53_tiling_closure CLOSURE CALL
  _pert_blocks3bnbias_mode0__dual53_tiling_closure(&DeeployNetwork__pert_blocks3bnbias_mode0__dual53_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr;
} _pert_blocks3bnbias_mode0__dual53_closure_args_t;

static void _pert_blocks3bnbias_mode0__dual53_closure(void *_pert_blocks3bnbias_mode0__dual53_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual53_closure_args_t *args = (_pert_blocks3bnbias_mode0__dual53_closure_args_t *)_pert_blocks3bnbias_mode0__dual53_closure_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_input_9_ref =
      (void *)((char *)DeeployNetwork_input_9 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_blocks3bnbias_pert_pm__dual52_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_input_9_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks3bnbias_mode0__dual53_cluster_fork_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual53_cluster_fork_args =
        (_pert_blocks3bnbias_mode0__dual53_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnbias_mode0__dual53_cluster_fork, &DeeployNetwork__pert_blocks3bnbias_mode0__dual53_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_blocks3bnbias_pert_pm__dual52_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr;
} _pert_blocks3bnbias_mode0__dual53_closure_L3_args_t;

static void _pert_blocks3bnbias_mode0__dual53_closure_L3(void *_pert_blocks3bnbias_mode0__dual53_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual53_closure_L3_args_t *args =
      (_pert_blocks3bnbias_mode0__dual53_closure_L3_args_t *)_pert_blocks3bnbias_mode0__dual53_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_mode0__dual53_closure_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual53_closure_args =
      (_pert_blocks3bnbias_mode0__dual53_closure_args_t){.DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor =
                                                             DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr};

  // _pert_blocks3bnbias_mode0__dual53_closure CLOSURE CALL
  _pert_blocks3bnbias_mode0__dual53_closure(&DeeployNetwork__pert_blocks3bnbias_mode0__dual53_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref;
} _pert_blocks4bnweight_mode0__dual58_tiling_closure_args_t;

static void _pert_blocks4bnweight_mode0__dual58_tiling_closure(void *_pert_blocks4bnweight_mode0__dual58_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual58_tiling_closure_args_t *args =
      (_pert_blocks4bnweight_mode0__dual58_tiling_closure_args_t *)_pert_blocks4bnweight_mode0__dual58_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks4bnweight_mode0__dual58, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks4bnweight_mode0__dual58_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnweight_mode0__dual58_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnweight_mode0__dual58_chunk = (32 >> pert_blocks4bnweight_mode0__dual58_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnweight_mode0__dual58_chunk_start =
      (uint32_t)MIN(pert_blocks4bnweight_mode0__dual58_chunk * pert_blocks4bnweight_mode0__dual58_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnweight_mode0__dual58_chunk_stop =
      (uint32_t)MIN(pert_blocks4bnweight_mode0__dual58_chunk_start + pert_blocks4bnweight_mode0__dual58_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnweight_mode0__dual58_local_size = pert_blocks4bnweight_mode0__dual58_chunk_stop - pert_blocks4bnweight_mode0__dual58_chunk_start;
  uint32_t pert_blocks4bnweight_mode0__dual58_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 20 + pert_blocks4bnweight_mode0__dual58_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnweight_mode0__dual58_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref[pert_blocks4bnweight_mode0__dual58_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref)[pert_blocks4bnweight_mode0__dual58_chunk_start],
      pert_blocks4bnweight_mode0__dual58_seed, 1u, pert_blocks4bnweight_mode0__dual58_local_size, pert_blocks4bnweight_mode0__dual58_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref[pert_blocks4bnweight_mode0__dual58_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref[32 + pert_blocks4bnweight_mode0__dual58_chunk_start],
      pert_blocks4bnweight_mode0__dual58_seed, 0u, pert_blocks4bnweight_mode0__dual58_local_size, pert_blocks4bnweight_mode0__dual58_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref;
} _pert_blocks4bnweight_mode0__dual58_cluster_fork_args_t;

static void _pert_blocks4bnweight_mode0__dual58_cluster_fork(void *_pert_blocks4bnweight_mode0__dual58_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual58_cluster_fork_args_t *args =
      (_pert_blocks4bnweight_mode0__dual58_cluster_fork_args_t *)_pert_blocks4bnweight_mode0__dual58_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_mode0__dual58_tiling_closure_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual58_tiling_closure_args =
      (_pert_blocks4bnweight_mode0__dual58_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref};

  // _pert_blocks4bnweight_mode0__dual58_tiling_closure CLOSURE CALL
  _pert_blocks4bnweight_mode0__dual58_tiling_closure(&DeeployNetwork__pert_blocks4bnweight_mode0__dual58_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr;
} _pert_blocks4bnweight_mode0__dual58_closure_args_t;

static void _pert_blocks4bnweight_mode0__dual58_closure(void *_pert_blocks4bnweight_mode0__dual58_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual58_closure_args_t *args =
      (_pert_blocks4bnweight_mode0__dual58_closure_args_t *)_pert_blocks4bnweight_mode0__dual58_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_input_10_ref =
      (void *)((char *)DeeployNetwork_input_10 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_blocks4bnweight_pert_pm__dual57_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_input_10_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks4bnweight_mode0__dual58_cluster_fork_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual58_cluster_fork_args =
        (_pert_blocks4bnweight_mode0__dual58_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnweight_mode0__dual58_cluster_fork, &DeeployNetwork__pert_blocks4bnweight_mode0__dual58_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_blocks4bnweight_pert_pm__dual57_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr;
} _pert_blocks4bnweight_mode0__dual58_closure_L3_args_t;

static void _pert_blocks4bnweight_mode0__dual58_closure_L3(void *_pert_blocks4bnweight_mode0__dual58_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual58_closure_L3_args_t *args =
      (_pert_blocks4bnweight_mode0__dual58_closure_L3_args_t *)_pert_blocks4bnweight_mode0__dual58_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_mode0__dual58_closure_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual58_closure_args =
      (_pert_blocks4bnweight_mode0__dual58_closure_args_t){.DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor =
                                                               DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr};

  // _pert_blocks4bnweight_mode0__dual58_closure CLOSURE CALL
  _pert_blocks4bnweight_mode0__dual58_closure(&DeeployNetwork__pert_blocks4bnweight_mode0__dual58_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref;
} _pert_blocks4bnbias_mode0__dual60_tiling_closure_args_t;

static void _pert_blocks4bnbias_mode0__dual60_tiling_closure(void *_pert_blocks4bnbias_mode0__dual60_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual60_tiling_closure_args_t *args =
      (_pert_blocks4bnbias_mode0__dual60_tiling_closure_args_t *)_pert_blocks4bnbias_mode0__dual60_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks4bnbias_mode0__dual60, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks4bnbias_mode0__dual60_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnbias_mode0__dual60_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnbias_mode0__dual60_chunk = (32 >> pert_blocks4bnbias_mode0__dual60_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnbias_mode0__dual60_chunk_start =
      (uint32_t)MIN(pert_blocks4bnbias_mode0__dual60_chunk * pert_blocks4bnbias_mode0__dual60_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnbias_mode0__dual60_chunk_stop =
      (uint32_t)MIN(pert_blocks4bnbias_mode0__dual60_chunk_start + pert_blocks4bnbias_mode0__dual60_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnbias_mode0__dual60_local_size = pert_blocks4bnbias_mode0__dual60_chunk_stop - pert_blocks4bnbias_mode0__dual60_chunk_start;
  uint32_t pert_blocks4bnbias_mode0__dual60_seed = ((42 + perturb_seed_base) + NUM_CORES * 21 + pert_blocks4bnbias_mode0__dual60_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnbias_mode0__dual60_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref[pert_blocks4bnbias_mode0__dual60_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref)[pert_blocks4bnbias_mode0__dual60_chunk_start],
      pert_blocks4bnbias_mode0__dual60_seed, 1u, pert_blocks4bnbias_mode0__dual60_local_size, pert_blocks4bnbias_mode0__dual60_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref[pert_blocks4bnbias_mode0__dual60_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref[32 + pert_blocks4bnbias_mode0__dual60_chunk_start],
      pert_blocks4bnbias_mode0__dual60_seed, 0u, pert_blocks4bnbias_mode0__dual60_local_size, pert_blocks4bnbias_mode0__dual60_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref;
} _pert_blocks4bnbias_mode0__dual60_cluster_fork_args_t;

static void _pert_blocks4bnbias_mode0__dual60_cluster_fork(void *_pert_blocks4bnbias_mode0__dual60_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual60_cluster_fork_args_t *args =
      (_pert_blocks4bnbias_mode0__dual60_cluster_fork_args_t *)_pert_blocks4bnbias_mode0__dual60_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_mode0__dual60_tiling_closure_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual60_tiling_closure_args =
      (_pert_blocks4bnbias_mode0__dual60_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref};

  // _pert_blocks4bnbias_mode0__dual60_tiling_closure CLOSURE CALL
  _pert_blocks4bnbias_mode0__dual60_tiling_closure(&DeeployNetwork__pert_blocks4bnbias_mode0__dual60_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr;
} _pert_blocks4bnbias_mode0__dual60_closure_args_t;

static void _pert_blocks4bnbias_mode0__dual60_closure(void *_pert_blocks4bnbias_mode0__dual60_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual60_closure_args_t *args = (_pert_blocks4bnbias_mode0__dual60_closure_args_t *)_pert_blocks4bnbias_mode0__dual60_closure_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_input_11_ref =
      (void *)((char *)DeeployNetwork_input_11 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_blocks4bnbias_pert_pm__dual59_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_input_11_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks4bnbias_mode0__dual60_cluster_fork_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual60_cluster_fork_args =
        (_pert_blocks4bnbias_mode0__dual60_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnbias_mode0__dual60_cluster_fork, &DeeployNetwork__pert_blocks4bnbias_mode0__dual60_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_blocks4bnbias_pert_pm__dual59_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr;
} _pert_blocks4bnbias_mode0__dual60_closure_L3_args_t;

static void _pert_blocks4bnbias_mode0__dual60_closure_L3(void *_pert_blocks4bnbias_mode0__dual60_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual60_closure_L3_args_t *args =
      (_pert_blocks4bnbias_mode0__dual60_closure_L3_args_t *)_pert_blocks4bnbias_mode0__dual60_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_mode0__dual60_closure_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual60_closure_args =
      (_pert_blocks4bnbias_mode0__dual60_closure_args_t){.DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor =
                                                             DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr};

  // _pert_blocks4bnbias_mode0__dual60_closure CLOSURE CALL
  _pert_blocks4bnbias_mode0__dual60_closure(&DeeployNetwork__pert_blocks4bnbias_mode0__dual60_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref;
} _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args_t;

static void _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure(void *_wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args_t *)_wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: wrappedInnerForwardImplConv_ne16wenc__dual3, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1
  // weight -- QW exp6
  {
    uint32_t wrappedInnerForwardImplConv_ne16wenc__dual3_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
    uint32_t wrappedInnerForwardImplConv_ne16wenc__dual3_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImplConv_ne16wenc__dual3_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref,
                               (int8_t *)DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer, wrappedInnerForwardImplConv_ne16wenc__dual3_seed_w,
                               wrappedInnerForwardImplConv_ne16wenc__dual3_tile_xor, 15, wrappedInnerForwardImplConv_ne16wenc__dual3_eps_scale, NUM_CORES, 8, 4,
                               0, 0, 1, 1, 4, (uint32_t *)DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref;
} _wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args_t;

static void _wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork(void *_wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args_t *)_wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer = DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref};

  // _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure(&DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr;
} _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args_t;

static void _wrappedInnerForwardImplConv_ne16wenc__dual3_closure(void *_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args_t *args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args_t *)_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args;

  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = args->DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 352);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_input_12_ref =
      (void *)((char *)DeeployNetwork_input_12 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16wenc__dual3_ref =
          (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16wenc__dual3 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_blocks0convweight_int8_ne16enc__dual1_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 288);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_input_12_ref);
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImplConv_ne16wenc__dual3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args =
        (_wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer = DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_blocks0convweight_int8_ne16enc__dual1_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr;
} _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args_t;

static void _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3(void *_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args_t *)_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args;

  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = args->DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args =
      (_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args_t){
          .DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr};

  // _wrappedInnerForwardImplConv_ne16wenc__dual3_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_ne16wenc__dual3_closure(&DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref;
} _pert_fcweight_fp32_mode1__dual69_tiling_closure_args_t;

static void _pert_fcweight_fp32_mode1__dual69_tiling_closure(void *_pert_fcweight_fp32_mode1__dual69_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual69_tiling_closure_args_t *args =
      (_pert_fcweight_fp32_mode1__dual69_tiling_closure_args_t *)_pert_fcweight_fp32_mode1__dual69_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcweight_fp32_mode1__dual69, Op: PerturbRademacherFixed, mode 1) -- QW exp8
  uint8_t pert_fcweight_fp32_mode1__dual69_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcweight_fp32_mode1__dual69_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcweight_fp32_mode1__dual69_chunk = (288 >> pert_fcweight_fp32_mode1__dual69_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcweight_fp32_mode1__dual69_chunk_start =
      (uint32_t)MIN(pert_fcweight_fp32_mode1__dual69_chunk * pert_fcweight_fp32_mode1__dual69_core_id, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode1__dual69_chunk_stop =
      (uint32_t)MIN(pert_fcweight_fp32_mode1__dual69_chunk_start + pert_fcweight_fp32_mode1__dual69_chunk, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode1__dual69_local_size = pert_fcweight_fp32_mode1__dual69_chunk_stop - pert_fcweight_fp32_mode1__dual69_chunk_start;
  uint32_t pert_fcweight_fp32_mode1__dual69_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_fcweight_fp32_mode1__dual69_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcweight_fp32_mode1__dual69_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref[pert_fcweight_fp32_mode1__dual69_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref)[pert_fcweight_fp32_mode1__dual69_chunk_start],
      pert_fcweight_fp32_mode1__dual69_seed, 1u, pert_fcweight_fp32_mode1__dual69_local_size, pert_fcweight_fp32_mode1__dual69_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref;
} _pert_fcweight_fp32_mode1__dual69_cluster_fork_args_t;

static void _pert_fcweight_fp32_mode1__dual69_cluster_fork(void *_pert_fcweight_fp32_mode1__dual69_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual69_cluster_fork_args_t *args =
      (_pert_fcweight_fp32_mode1__dual69_cluster_fork_args_t *)_pert_fcweight_fp32_mode1__dual69_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode1__dual69_tiling_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual69_tiling_closure_args =
      (_pert_fcweight_fp32_mode1__dual69_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref};

  // _pert_fcweight_fp32_mode1__dual69_tiling_closure CLOSURE CALL
  _pert_fcweight_fp32_mode1__dual69_tiling_closure(&DeeployNetwork__pert_fcweight_fp32_mode1__dual69_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr;
} _pert_fcweight_fp32_mode1__dual69_closure_args_t;

static void _pert_fcweight_fp32_mode1__dual69_closure(void *_pert_fcweight_fp32_mode1__dual69_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual69_closure_args_t *args = (_pert_fcweight_fp32_mode1__dual69_closure_args_t *)_pert_fcweight_fp32_mode1__dual69_closure_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_input_22_ref =
      (void *)((char *)DeeployNetwork_input_22 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_fcweight_fp32_pert_p__dual68_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_input_22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcweight_fp32_mode1__dual69_cluster_fork_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual69_cluster_fork_args =
        (_pert_fcweight_fp32_mode1__dual69_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcweight_fp32_mode1__dual69_cluster_fork, &DeeployNetwork__pert_fcweight_fp32_mode1__dual69_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_fcweight_fp32_pert_p__dual68_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr;
} _pert_fcweight_fp32_mode1__dual69_closure_L3_args_t;

static void _pert_fcweight_fp32_mode1__dual69_closure_L3(void *_pert_fcweight_fp32_mode1__dual69_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual69_closure_L3_args_t *args =
      (_pert_fcweight_fp32_mode1__dual69_closure_L3_args_t *)_pert_fcweight_fp32_mode1__dual69_closure_L3_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode1__dual69_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual69_closure_args =
      (_pert_fcweight_fp32_mode1__dual69_closure_args_t){.DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor =
                                                             DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr};

  // _pert_fcweight_fp32_mode1__dual69_closure CLOSURE CALL
  _pert_fcweight_fp32_mode1__dual69_closure(&DeeployNetwork__pert_fcweight_fp32_mode1__dual69_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref;
} _pert_fcweight_fp32_mode2__dual75_tiling_closure_args_t;

static void _pert_fcweight_fp32_mode2__dual75_tiling_closure(void *_pert_fcweight_fp32_mode2__dual75_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual75_tiling_closure_args_t *args =
      (_pert_fcweight_fp32_mode2__dual75_tiling_closure_args_t *)_pert_fcweight_fp32_mode2__dual75_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcweight_fp32_mode2__dual75, Op: PerturbRademacherFixed, mode 2) -- QW exp8
  uint8_t pert_fcweight_fp32_mode2__dual75_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcweight_fp32_mode2__dual75_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcweight_fp32_mode2__dual75_chunk = (288 >> pert_fcweight_fp32_mode2__dual75_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcweight_fp32_mode2__dual75_chunk_start =
      (uint32_t)MIN(pert_fcweight_fp32_mode2__dual75_chunk * pert_fcweight_fp32_mode2__dual75_core_id, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode2__dual75_chunk_stop =
      (uint32_t)MIN(pert_fcweight_fp32_mode2__dual75_chunk_start + pert_fcweight_fp32_mode2__dual75_chunk, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode2__dual75_local_size = pert_fcweight_fp32_mode2__dual75_chunk_stop - pert_fcweight_fp32_mode2__dual75_chunk_start;
  uint32_t pert_fcweight_fp32_mode2__dual75_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_fcweight_fp32_mode2__dual75_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcweight_fp32_mode2__dual75_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref[pert_fcweight_fp32_mode2__dual75_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref)[pert_fcweight_fp32_mode2__dual75_chunk_start],
      pert_fcweight_fp32_mode2__dual75_seed, 0u, pert_fcweight_fp32_mode2__dual75_local_size, pert_fcweight_fp32_mode2__dual75_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref;
} _pert_fcweight_fp32_mode2__dual75_cluster_fork_args_t;

static void _pert_fcweight_fp32_mode2__dual75_cluster_fork(void *_pert_fcweight_fp32_mode2__dual75_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual75_cluster_fork_args_t *args =
      (_pert_fcweight_fp32_mode2__dual75_cluster_fork_args_t *)_pert_fcweight_fp32_mode2__dual75_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode2__dual75_tiling_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual75_tiling_closure_args =
      (_pert_fcweight_fp32_mode2__dual75_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref};

  // _pert_fcweight_fp32_mode2__dual75_tiling_closure CLOSURE CALL
  _pert_fcweight_fp32_mode2__dual75_tiling_closure(&DeeployNetwork__pert_fcweight_fp32_mode2__dual75_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr;
} _pert_fcweight_fp32_mode2__dual75_closure_args_t;

static void _pert_fcweight_fp32_mode2__dual75_closure(void *_pert_fcweight_fp32_mode2__dual75_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual75_closure_args_t *args = (_pert_fcweight_fp32_mode2__dual75_closure_args_t *)_pert_fcweight_fp32_mode2__dual75_closure_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_input_22_ref =
      (void *)((char *)DeeployNetwork_input_22 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_fcweight_fp32_pert_m__dual74_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_input_22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcweight_fp32_mode2__dual75_cluster_fork_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual75_cluster_fork_args =
        (_pert_fcweight_fp32_mode2__dual75_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcweight_fp32_mode2__dual75_cluster_fork, &DeeployNetwork__pert_fcweight_fp32_mode2__dual75_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_fcweight_fp32_pert_m__dual74_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr;
} _pert_fcweight_fp32_mode2__dual75_closure_L3_args_t;

static void _pert_fcweight_fp32_mode2__dual75_closure_L3(void *_pert_fcweight_fp32_mode2__dual75_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual75_closure_L3_args_t *args =
      (_pert_fcweight_fp32_mode2__dual75_closure_L3_args_t *)_pert_fcweight_fp32_mode2__dual75_closure_L3_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode2__dual75_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual75_closure_args =
      (_pert_fcweight_fp32_mode2__dual75_closure_args_t){.DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor =
                                                             DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr};

  // _pert_fcweight_fp32_mode2__dual75_closure CLOSURE CALL
  _pert_fcweight_fp32_mode2__dual75_closure(&DeeployNetwork__pert_fcweight_fp32_mode2__dual75_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure(void *_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: wrappedInnerForwardImpl_1Conv_ne16wenc__dual21, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1
  // weight -- QW exp6
  {
    uint32_t wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
    uint32_t wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref,
                               (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer,
                               wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_seed_w, wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tile_xor, 15,
                               wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_eps_scale, NUM_CORES, 16, 128, 0, 1, 8, 1, 16,
                               (uint32_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork(void *_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args_t *)_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure(void *_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args;

  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = args->DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref =
      (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10240);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref =
      (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6144);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_input_13_ref =
      (void *)((char *)DeeployNetwork_input_13 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_ref =
          (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_blocks1convweight_int8_ne16enc__dual19_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_input_13_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args =
        (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314816, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_blocks1convweight_int8_ne16enc__dual19_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3(void *_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args_t *)_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args;

  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = args->DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args =
      (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args_t){
          .DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref;
} _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure(void *_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: wrappedInnerForwardImpl_2Conv_ne16wenc__dual36, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1
  // weight -- QW exp6
  {
    uint32_t wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 4);
    uint32_t wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tile_xor = (0 * 0x9E3779B1u);
    float32_t wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref,
                               (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer,
                               wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_seed_w, wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tile_xor, 15,
                               wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_eps_scale, NUM_CORES, 16, 128, 0, 1, 16, 1, 8,
                               (uint32_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_x21c_hcache);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref;
} _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork(void *_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args_t *)_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer = DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref};

  // _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure(void *_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args;

  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = args->DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref =
      (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10240);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref =
      (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6144);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_input_14_ref =
      (void *)((char *)DeeployNetwork_input_14 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_ref =
          (void *)((char *)DeeployNetwork_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36 +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_blocks2convweight_int8_ne16enc__dual34_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_input_14_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_blocks2convweight_int8_pmul_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args =
        (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer = DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314816, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_blocks2convweight_int8_ne16enc__dual34_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3(void *_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args_t *)_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args;

  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = args->DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args =
      (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args_t){
          .DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr};

  // _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref;
} _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args_t;

static void _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure(void *_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args) {
  // CLOSURE ARG CAST
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args_t *args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args_t *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Quantization (Name: QCDQ_blocks_0_conv_input_quant_1Clip_Quant, Op: Quant)
  uint8_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_core_id = (uint8_t)pi_core_id();
  uint8_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk = (9800 >> QCDQ_blocks_0_conv_input_quant_1Clip_Quant_log2Core) + ((9800 & (NUM_CORES - 1)) != 0);
  uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk_start =
      (uint32_t)MIN(QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk * QCDQ_blocks_0_conv_input_quant_1Clip_Quant_core_id, (uint32_t)9800);
  uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk_stop =
      (uint32_t)MIN(QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk_start + QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk, (uint32_t)9800);

  for (uint32_t i = QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk_start; i < QCDQ_blocks_0_conv_input_quant_1Clip_Quant_chunk_stop; i++) {
    // quantization formula
    float32_t input_val = DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref[i];
    float32_t scaled_val =
        input_val * ((float32_t)0.09778456837280367); // Multiply instead of divide -- QW: fp32 cast (bare literal is double -> soft-float on fp32-only FPU)
    float32_t shifted_val = scaled_val + ((float32_t)0.0); // -- QW fp32 cast

    // Round to nearest integer
    int32_t quantized = (int32_t)(shifted_val + 0.5f * (shifted_val >= 0 ? 1 : -1));

    // Clamp the value
    if (quantized < -128)
      quantized = -128;
    if (quantized > 127)
      quantized = 127;

    // Assign directly with explicit cast
    DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref[i] = (int8_t)quantized;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref;
} _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t;

static void _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork(void *_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args) {
  // CLOSURE ARG CAST
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t *args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref;

  // CLOSURE FUNCTION CALL
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref};

  // _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure CLOSURE CALL
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure(&DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;
} _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args_t;

static void _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure(void *_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args) {
  // CLOSURE ARG CAST
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args_t *args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args_t *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39200);
  void *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1480992, DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args =
        (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork,
                    &DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1320520, DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;
} _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t;

static void _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3(void *_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args) {
  // CLOSURE ARG CAST
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t *args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args =
      (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args_t){.DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor =
                                                                       DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
                                                                   .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr};

  // _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure CLOSURE CALL
  _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure(&DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref;
} _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args_t;

static void _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure(void *_wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args_t *)_wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: wrappedInnerForwardImplConv_im2col__dual7, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref, NUM_CORES, 14, 700, 1, 14, 701,
                            1, 4, 0, 0, 2, 2, 1, 1, (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref;
} _wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args_t;

static void _wrappedInnerForwardImplConv_im2col__dual7_cluster_fork(void *_wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args_t *)_wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args =
      (_wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref};

  // _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_im2col__dual7_tiling_closure(&DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr;
} _wrappedInnerForwardImplConv_im2col__dual7_closure_args_t;

static void _wrappedInnerForwardImplConv_im2col__dual7_closure(void *_wrappedInnerForwardImplConv_im2col__dual7_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_im2col__dual7_closure_args_t *args =
      (_wrappedInnerForwardImplConv_im2col__dual7_closure_args_t *)_wrappedInnerForwardImplConv_im2col__dual7_closure_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor =
      args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39256);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451592, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args =
        (_wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv_im2col__dual7_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1349976, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr;
} _wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args_t;

static void _wrappedInnerForwardImplConv_im2col__dual7_closure_L3(void *_wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args_t *)_wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor =
      args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv_im2col__dual7_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_closure_args =
      (_wrappedInnerForwardImplConv_im2col__dual7_closure_args_t){.DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor =
                                                                      DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
                                                                  .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor =
                                                                      DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr};

  // _wrappedInnerForwardImplConv_im2col__dual7_closure CLOSURE CALL
  _wrappedInnerForwardImplConv_im2col__dual7_closure(&DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_tiling_closure(void *__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // N-EUREKA Task Init
  ne16_task_t task = {
      .data = (ne16_task_data_t){
          .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref - 0 + 0,
          .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref - 0,
          .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
          .scale_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .scale_shift_addr = (uint32_t)NULL,
          .scale_bias_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
          .cfg = (ne16_cfg_t){
              .input_stride = (ne16_stride_t){.d0 = 4, .d1 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                                               .d1 = 16,
                                               .d2 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                     .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref,
                                                                             *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref)},
                                          .remainder = {.KoKi = nnx_concat_half(16, 4),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref)}},
              .padding = (0 << 28) + (0 << 24) + (0 << 20) + (0 << 16),
              .weight_offset_factor = -128,
              .filter_mask = 0,
              .conf0 = 43032663,
          }}};
  // NE16 top-level task struct fields (required by HAL helpers and NE16 HW for
  // non-1x1 paths). Kept consistent with ne16_task_set_op_to_conv/_set_bits.
  task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
  task.qw = 8;
  task.subtile_output_channel = 32;
  task.kernel_shape = 1;
  task.depthwise = 0;

  // N-EUREKA Task Execution
  ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
  ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
  ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = args->DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped =
      args->DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 90432);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113040);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113360);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113296);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped_ref =
          (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_ne16enc__dual1_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_x2__dual8_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_x2__dual8_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_ne16add2__dual2_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);

  // Initialize DMA futures
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
          .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[TILING_I],
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped_ref,
          .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[TILING_I],
          .ext_stride_1d = 2804};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped_ref) +
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[TILING_I]);

    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_ne16enc__dual1_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_x2__dual8_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_ne16add2__dual2_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride[TILING_I];

    __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    // __MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
          .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[TILING_I],
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_ref,
          .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[TILING_I],
          .ext_stride_1d = 11216};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = args->DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped =
      args->DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor,
      .DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor,
      .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped =
          DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped,
      .DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped =
          DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref;
} _bnBatchNormalization_tail_tiling_closure_args_t;

static void _bnBatchNormalization_tail_tiling_closure(void *_bnBatchNormalization_tail_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_tiling_closure_args_t *args = (_bnBatchNormalization_tail_tiling_closure_args_t *)_bnBatchNormalization_tail_tiling_closure_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL

  // QW exp9: frozen BN + MaxPool + Relu + Quant, int8 HWC -> int8 HWC (Name: bnBatchNormalization_tail, Op: FrozenBNPoolReluQuant)
  PULP_FrozenBNPoolReluQuantHWC_i8_i8(
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref,
      *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref, 701, 87, 16, 1, 8, 9.999999747378752e-06f, (float32_t)6.1015625, (float32_t)0.0,
      (float32_t)8.295696202531646, (float32_t)0.0, -128, 127);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref;
} _bnBatchNormalization_tail_cluster_fork_args_t;

static void _bnBatchNormalization_tail_cluster_fork(void *_bnBatchNormalization_tail_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_cluster_fork_args_t *args = (_bnBatchNormalization_tail_cluster_fork_args_t *)_bnBatchNormalization_tail_cluster_fork_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL
  _bnBatchNormalization_tail_tiling_closure_args_t DeeployNetwork__bnBatchNormalization_tail_tiling_closure_args =
      (_bnBatchNormalization_tail_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref};

  // _bnBatchNormalization_tail_tiling_closure CLOSURE CALL
  _bnBatchNormalization_tail_tiling_closure(&DeeployNetwork__bnBatchNormalization_tail_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_args_t;

static void _bnBatchNormalization_tail_closure(void *_bnBatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_args_t *args = (_bnBatchNormalization_tail_closure_args_t *)_bnBatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor =
      args->DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
      args->DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113472);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113536);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113600);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113664);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 100944);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_pm__dual13_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_pm__dual15_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean__dual17_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_mean__dual17_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var__dual18_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_var__dual18_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_cumByteOffset
                                                                                [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cmd[TILING_I],
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_ref);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_ref) + 100944);

    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_pm__dual13_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_pm__dual15_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean__dual17_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var__dual18_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref
    DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref = &DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out[TILING_I];

    _bnBatchNormalization_tail_cluster_fork_args_t DeeployNetwork__bnBatchNormalization_tail_cluster_fork_args =
        (_bnBatchNormalization_tail_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_H_out_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bnBatchNormalization_tail_cluster_fork, &DeeployNetwork__bnBatchNormalization_tail_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cmd[TILING_I],
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref) + 12528);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_L3_args_t;

static void _bnBatchNormalization_tail_closure_L3(void *_bnBatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_L3_args_t *args = (_bnBatchNormalization_tail_closure_L3_args_t *)_bnBatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor =
      args->DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
      args->DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bnBatchNormalization_tail_closure_args_t DeeployNetwork__bnBatchNormalization_tail_closure_args = (_bnBatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor,
      .DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor,
      .DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor = DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor,
      .DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor = DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

  // _bnBatchNormalization_tail_closure CLOSURE CALL
  _bnBatchNormalization_tail_closure(&DeeployNetwork__bnBatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure(void *_wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualConv2in, stacked [x+ | x-] input (Name: wrappedInnerForwardImpl_1Conv_ne16__dual26, Op: NE16DualConv2in), pipeline=1 -- QW exp8
  PULP_NE16DualConv2in((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref,
                       (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref + 8,
                       (const uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref,
                       (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref,
                       (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref,
                       (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref,
                       (uint8_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf, NUM_CORES, 43032663u, 14, 87, 8, 14, 88, 16, 1, 16, 0, 0, 8,
                       8, 1, 1, -128, 1, 27, 2 * 8, 0);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure(void *_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args_t *)_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = args->DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
      args->DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
      args->DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 60160);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 79648);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 83872);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 83744);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks1convweight_int8_ne16enc__dual19_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16__dual26_ref =
      (void
           *)((char *)
                  DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16__dual26 +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks1convbias_rqsadd_ne16add2__dual20_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39424);

  // Initialize DMA futures
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1461280,
          .size = 19488,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445888,
          .size = 4096,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks1convweight_int8_ne16enc__dual19_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_1Conv_ne16__dual26_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks1convbias_rqsadd_ne16add2__dual20_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args =
        (_wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf = DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref};

    // _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1350144,
          .size = 39424,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_data_out_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3(void *_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args_t *)_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = args->DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
      args->DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
      args->DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args =
      (_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args_t){
          .DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor,
          .DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor,
          .DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
              DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor,
          .DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
              DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref;
} _bn_1BatchNormalization_tail_tiling_closure_args_t;

static void _bn_1BatchNormalization_tail_tiling_closure(void *_bn_1BatchNormalization_tail_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_tiling_closure_args_t *args =
      (_bn_1BatchNormalization_tail_tiling_closure_args_t *)_bn_1BatchNormalization_tail_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL

  // QW exp9: frozen BN + MaxPool + Relu + Quant, int8 HWC -> int8 HWC (Name: bn_1BatchNormalization_tail, Op: FrozenBNPoolReluQuant)
  PULP_FrozenBNPoolReluQuantHWC_i8_i8(
      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref,
      14, 88, 22, 32, 1, 4, 9.999999747378752e-06f, (float32_t)0.1591796875, (float32_t)0.0, (float32_t)15.398496240601503, (float32_t)0.0, -128, 127);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref;
} _bn_1BatchNormalization_tail_cluster_fork_args_t;

static void _bn_1BatchNormalization_tail_cluster_fork(void *_bn_1BatchNormalization_tail_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_cluster_fork_args_t *args = (_bn_1BatchNormalization_tail_cluster_fork_args_t *)_bn_1BatchNormalization_tail_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL
  _bn_1BatchNormalization_tail_tiling_closure_args_t DeeployNetwork__bn_1BatchNormalization_tail_tiling_closure_args =
      (_bn_1BatchNormalization_tail_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref};

  // _bn_1BatchNormalization_tail_tiling_closure CLOSURE CALL
  _bn_1BatchNormalization_tail_tiling_closure(&DeeployNetwork__bn_1BatchNormalization_tail_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_args_t;

static void _bn_1BatchNormalization_tail_closure(void *_bn_1BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_args_t *args = (_bn_1BatchNormalization_tail_closure_args_t *)_bn_1BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
      args->DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
      args->DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49408);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49536);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49664);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39424);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_pm__dual28_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_pm__dual30_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean__dual32_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_mean__dual32_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var__dual33_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_var__dual33_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1481216, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_pm__dual28_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_pm__dual30_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean__dual32_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var__dual33_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _bn_1BatchNormalization_tail_cluster_fork_args_t DeeployNetwork__bn_1BatchNormalization_tail_cluster_fork_args =
        (_bn_1BatchNormalization_tail_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_1BatchNormalization_tail_cluster_fork, &DeeployNetwork__bn_1BatchNormalization_tail_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1320576, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_L3_args_t;

static void _bn_1BatchNormalization_tail_closure_L3(void *_bn_1BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_L3_args_t *args = (_bn_1BatchNormalization_tail_closure_L3_args_t *)_bn_1BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
      args->DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
      args->DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_1BatchNormalization_tail_closure_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_args = (_bn_1BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor,
      .DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor,
      .DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
          DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor,
      .DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor = DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

  // _bn_1BatchNormalization_tail_closure CLOSURE CALL
  _bn_1BatchNormalization_tail_closure(&DeeployNetwork__bn_1BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref;
} _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure(void *_wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualConv2in, stacked [x+ | x-] input (Name: wrappedInnerForwardImpl_2Conv_ne16__dual41, Op: NE16DualConv2in), pipeline=1 -- QW exp8
  PULP_NE16DualConv2in((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref,
                       (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref + 16,
                       (const uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref,
                       (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref,
                       (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref,
                       (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref,
                       (uint8_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf, NUM_CORES, 43032663u, 14, 22, 16, 14, 23, 16, 1, 8, 0, 0, 4,
                       4, 1, 1, -128, 1, 27, 2 * 16, 0);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure(void *_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args_t *)_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = args->DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
      args->DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
      args->DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31040);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 40896);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 45120);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 44992);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 20736);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks2convweight_int8_ne16enc__dual34_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16__dual41_ref =
      (void
           *)((char *)
                  DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16__dual41 +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks2convbias_rqsadd_ne16add2__dual35_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1451648,
          .size = 9856,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445888,
          .size = 4096,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks2convweight_int8_ne16enc__dual34_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_wrappedInnerForwardImpl_2Conv_ne16__dual41_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks2convbias_rqsadd_ne16add2__dual35_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args =
        (_wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf = DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref};

    // _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure CLOSURE CALL
    _wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1321024,
          .size = 10304,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_data_out_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args_t;

static void _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3(void *_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args_t *)_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = args->DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
      args->DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
      args->DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args =
      (_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args_t){
          .DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor,
          .DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor,
          .DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
              DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor,
          .DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
              DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr};

  // _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref;
} _bn_2BatchNormalization_tail_tiling_closure_args_t;

static void _bn_2BatchNormalization_tail_tiling_closure(void *_bn_2BatchNormalization_tail_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_tiling_closure_args_t *args =
      (_bn_2BatchNormalization_tail_tiling_closure_args_t *)_bn_2BatchNormalization_tail_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL

  // QW exp9: frozen BN + MaxPool + Relu + Quant, int8 HWC -> int8 HWC (Name: bn_2BatchNormalization_tail, Op: FrozenBNPoolReluQuant)
  PULP_FrozenBNPoolReluQuantHWC_i8_i8(
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref,
      14, 23, 5, 32, 1, 4, 9.999999747378752e-06f, (float32_t)0.05462646484375, (float32_t)0.0, (float32_t)23.07605633802817, (float32_t)0.0, -128, 127);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref;
} _bn_2BatchNormalization_tail_cluster_fork_args_t;

static void _bn_2BatchNormalization_tail_cluster_fork(void *_bn_2BatchNormalization_tail_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_cluster_fork_args_t *args = (_bn_2BatchNormalization_tail_cluster_fork_args_t *)_bn_2BatchNormalization_tail_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL
  _bn_2BatchNormalization_tail_tiling_closure_args_t DeeployNetwork__bn_2BatchNormalization_tail_tiling_closure_args =
      (_bn_2BatchNormalization_tail_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref};

  // _bn_2BatchNormalization_tail_tiling_closure CLOSURE CALL
  _bn_2BatchNormalization_tail_tiling_closure(&DeeployNetwork__bn_2BatchNormalization_tail_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_args_t;

static void _bn_2BatchNormalization_tail_closure(void *_bn_2BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_args_t *args = (_bn_2BatchNormalization_tail_closure_args_t *)_bn_2BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
      args->DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12544);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12672);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12800);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12928);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10304);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_pm__dual43_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_pm__dual45_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean__dual47_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_mean__dual47_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var__dual48_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_var__dual48_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1452096, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_pm__dual43_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_pm__dual45_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean__dual47_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var__dual48_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _bn_2BatchNormalization_tail_cluster_fork_args_t DeeployNetwork__bn_2BatchNormalization_tail_cluster_fork_args =
        (_bn_2BatchNormalization_tail_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_2BatchNormalization_tail_cluster_fork, &DeeployNetwork__bn_2BatchNormalization_tail_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1312960, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_L3_args_t;

static void _bn_2BatchNormalization_tail_closure_L3(void *_bn_2BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_L3_args_t *args = (_bn_2BatchNormalization_tail_closure_L3_args_t *)_bn_2BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
      args->DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_2BatchNormalization_tail_closure_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_args = (_bn_2BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor,
      .DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor,
      .DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
          DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

  // _bn_2BatchNormalization_tail_closure CLOSURE CALL
  _bn_2BatchNormalization_tail_closure(&DeeployNetwork__bn_2BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref;
} _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure(void *_wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args_t *)_wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_3Conv__dual49, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_3Conv__dual49_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
  uint32_t wrappedInnerForwardImpl_3Conv__dual49_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 7);
  uint32_t wrappedInnerForwardImpl_3Conv__dual49_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_3Conv__dual49_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref + 16,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref + ((64) >> 1), (64), 32,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer, wrappedInnerForwardImpl_3Conv__dual49_seed_w,
                                   wrappedInnerForwardImpl_3Conv__dual49_seed_b, wrappedInnerForwardImpl_3Conv__dual49_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_3Conv__dual49_eps_scale, NUM_CORES, 16, 5, 14, 16, 5, 8, ((64) >> 1), 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref;
} _wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual49_cluster_fork(void *_wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args_t *)_wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args =
      (_wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref};

  // _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_3Conv__dual49_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr;
} _wrappedInnerForwardImpl_3Conv__dual49_closure_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual49_closure(void *_wrappedInnerForwardImpl_3Conv__dual49_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual49_closure_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual49_closure_args_t *)_wrappedInnerForwardImpl_3Conv__dual49_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 17280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19520);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19648);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19776);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14720);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_input_15_ref =
      (void *)((char *)DeeployNetwork_input_15 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_input_20_ref =
      (void *)((char *)DeeployNetwork_input_20 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_blocks3convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_blocks3convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3712);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1444032, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_input_15_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_input_20_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_blocks3convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_blocks3convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args =
        (_wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_3Conv__dual49_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1313280, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr;
} _wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual49_closure_L3(void *_wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args_t *)_wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_3Conv__dual49_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_closure_args =
      (_wrappedInnerForwardImpl_3Conv__dual49_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr};

  // _wrappedInnerForwardImpl_3Conv__dual49_closure CLOSURE CALL
  _wrappedInnerForwardImpl_3Conv__dual49_closure(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref;
} _bn_3BatchNormalization_tail_tiling_closure_args_t;

static void _bn_3BatchNormalization_tail_tiling_closure(void *_bn_3BatchNormalization_tail_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_tiling_closure_args_t *args =
      (_bn_3BatchNormalization_tail_tiling_closure_args_t *)_bn_3BatchNormalization_tail_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL

  // QW exp9: frozen BN + MaxPool + Relu + Quant, int8 HWC -> int8 HWC (Name: bn_3BatchNormalization_tail, Op: FrozenBNPoolReluQuant)
  PULP_FrozenBNPoolReluQuantHWC_i8_i8(
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref,
      8, 5, 5, 64, 1, 1, 9.999999747378752e-06f, (float32_t)0.04663088917732239, (float32_t)0.0, (float32_t)41.011264080100126, (float32_t)0.0, -128, 127);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref;
} _bn_3BatchNormalization_tail_cluster_fork_args_t;

static void _bn_3BatchNormalization_tail_cluster_fork(void *_bn_3BatchNormalization_tail_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_cluster_fork_args_t *args = (_bn_3BatchNormalization_tail_cluster_fork_args_t *)_bn_3BatchNormalization_tail_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref;

  // CLOSURE FUNCTION CALL
  _bn_3BatchNormalization_tail_tiling_closure_args_t DeeployNetwork__bn_3BatchNormalization_tail_tiling_closure_args =
      (_bn_3BatchNormalization_tail_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref};

  // _bn_3BatchNormalization_tail_tiling_closure CLOSURE CALL
  _bn_3BatchNormalization_tail_tiling_closure(&DeeployNetwork__bn_3BatchNormalization_tail_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_args_t;

static void _bn_3BatchNormalization_tail_closure(void *_bn_3BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_args_t *args = (_bn_3BatchNormalization_tail_closure_args_t *)_bn_3BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5376);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5120);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5632);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5888);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_pm__dual50_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_pm__dual52_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean__dual54_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_mean__dual54_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var__dual55_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_var__dual55_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1444352, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_pm__dual50_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_pm__dual52_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean__dual54_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var__dual55_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _bn_3BatchNormalization_tail_cluster_fork_args_t DeeployNetwork__bn_3BatchNormalization_tail_cluster_fork_args =
        (_bn_3BatchNormalization_tail_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_3BatchNormalization_tail_cluster_fork, &DeeployNetwork__bn_3BatchNormalization_tail_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1313280, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_L3_args_t;

static void _bn_3BatchNormalization_tail_closure_L3(void *_bn_3BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_L3_args_t *args = (_bn_3BatchNormalization_tail_closure_L3_args_t *)_bn_3BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_3BatchNormalization_tail_closure_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_args = (_bn_3BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor,
      .DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

  // _bn_3BatchNormalization_tail_closure CLOSURE CALL
  _bn_3BatchNormalization_tail_closure(&DeeployNetwork__bn_3BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref;
} _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure(void *_wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args_t *)_wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_4Conv__dual56, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_4Conv__dual56_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
  uint32_t wrappedInnerForwardImpl_4Conv__dual56_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
  uint32_t wrappedInnerForwardImpl_4Conv__dual56_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_4Conv__dual56_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref + 32,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref + ((64) >> 1), (64), 64,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer, wrappedInnerForwardImpl_4Conv__dual56_seed_w,
                                   wrappedInnerForwardImpl_4Conv__dual56_seed_b, wrappedInnerForwardImpl_4Conv__dual56_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_4Conv__dual56_eps_scale, NUM_CORES, 16, 5, 8, 32, 5, 2, ((64) >> 1), 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref;
} _wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual56_cluster_fork(void *_wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args_t *)_wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args =
      (_wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref};

  // _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv__dual56_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv__dual56_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual56_closure(void *_wrappedInnerForwardImpl_4Conv__dual56_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual56_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual56_closure_args_t *)_wrappedInnerForwardImpl_4Conv__dual56_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29056);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32256);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32384);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32512);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31616);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_input_16_ref =
      (void *)((char *)DeeployNetwork_input_16 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_input_21_ref =
      (void *)((char *)DeeployNetwork_input_21 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7296);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1444352, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_input_16_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_input_21_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_blocks4convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args =
        (_wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_4Conv__dual56_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311360, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual56_closure_L3(void *_wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args_t *)_wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv__dual56_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_closure_args =
      (_wrappedInnerForwardImpl_4Conv__dual56_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr};

  // _wrappedInnerForwardImpl_4Conv__dual56_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv__dual56_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref;
} _bn_4BatchNormalization_tiling_closure_args_t;

static void _bn_4BatchNormalization_tiling_closure(void *_bn_4BatchNormalization_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_tiling_closure_args_t *args = (_bn_4BatchNormalization_tiling_closure_args_t *)_bn_4BatchNormalization_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref;

  // CLOSURE FUNCTION CALL

  // BatchNormInternal (Name: bn_4BatchNormalization, Op: BatchNormInternal)
  // RW 2026-10-15: channels-last frozen-stat BN reading the int8 tensor of a folded Dequant (FoldDequantIntoFrozenBNPass)
  PULP_BatchNormInternalHWC_i8_fp32(
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref,
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref, DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref,
      1 * 2 * 5, 64, 9.999999747378752e-06f, (float32_t)0.0173187255859375, (float32_t)0.0);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref;
} _bn_4BatchNormalization_cluster_fork_args_t;

static void _bn_4BatchNormalization_cluster_fork(void *_bn_4BatchNormalization_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_cluster_fork_args_t *args = (_bn_4BatchNormalization_cluster_fork_args_t *)_bn_4BatchNormalization_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_tiling_closure_args_t DeeployNetwork__bn_4BatchNormalization_tiling_closure_args = (_bn_4BatchNormalization_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref};

  // _bn_4BatchNormalization_tiling_closure CLOSURE CALL
  _bn_4BatchNormalization_tiling_closure(&DeeployNetwork__bn_4BatchNormalization_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_args_t;

static void _bn_4BatchNormalization_closure(void *_bn_4BatchNormalization_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_args_t *args = (_bn_4BatchNormalization_closure_args_t *)_bn_4BatchNormalization_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3456);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3200);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3712);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3968);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4992);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4736);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_pm__dual57_tensor_ref =
      (void
           *)((char *)DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor +
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_pm__dual59_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean__dual61_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_mean__dual61_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var__dual62_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_var__dual62_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_cumByteOffset
                                                                                [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_sm_tensor_ref =
      (void *)((char *)DeeployNetwork_bn_4BatchNormalization_sm_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_cumByteOffset
                                                                             [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_siv_tensor_ref =
      (void *)((char *)DeeployNetwork_bn_4BatchNormalization_siv_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);

  // Initialize DMA futures
  int bias_ExternalToLocal = -1;
  int running_var_ExternalToLocal = -1;
  int saved_inv_std_LocalToExternal = -1;
  int data_out_LocalToExternal = -1;
  int running_mean_ExternalToLocal = -1;
  int scale_ExternalToLocal = -1;
  int saved_mean_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1442432,
          .size = 640,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    scale_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_pm__dual57_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    bias_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_pm__dual59_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_mean_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean__dual61_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_var_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var__dual62_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (running_var_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_var_ExternalToLocal);
      mchan_transfer_free(running_var_ExternalToLocal);
    }

    if (bias_ExternalToLocal >= 0) {
      mchan_transfer_wait(bias_ExternalToLocal);
      mchan_transfer_free(bias_ExternalToLocal);
    }

    if (running_mean_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_mean_ExternalToLocal);
      mchan_transfer_free(running_mean_ExternalToLocal);
    }

    if (scale_ExternalToLocal >= 0) {
      mchan_transfer_wait(scale_ExternalToLocal);
      mchan_transfer_free(scale_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _bn_4BatchNormalization_cluster_fork_args_t DeeployNetwork__bn_4BatchNormalization_cluster_fork_args = (_bn_4BatchNormalization_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_4BatchNormalization_cluster_fork, &DeeployNetwork__bn_4BatchNormalization_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1313280,
          .size = 2560,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    saved_mean_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310976,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_sm_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    saved_inv_std_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310976,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_siv_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    if (saved_mean_LocalToExternal >= 0) {
      mchan_transfer_wait(saved_mean_LocalToExternal);
      mchan_transfer_free(saved_mean_LocalToExternal);
    }

    if (saved_inv_std_LocalToExternal >= 0) {
      mchan_transfer_wait(saved_inv_std_LocalToExternal);
      mchan_transfer_free(saved_inv_std_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_L3_args_t;

static void _bn_4BatchNormalization_closure_L3(void *_bn_4BatchNormalization_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_L3_args_t *args = (_bn_4BatchNormalization_closure_L3_args_t *)_bn_4BatchNormalization_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_closure_args_t DeeployNetwork__bn_4BatchNormalization_closure_args = (_bn_4BatchNormalization_closure_args_t){
      .DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor,
      .DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
          DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed,
      .DeeployNetwork_bn_4BatchNormalization_sm_tensor = DeeployNetwork_bn_4BatchNormalization_sm_tensor,
      .DeeployNetwork_bn_4BatchNormalization_siv_tensor = DeeployNetwork_bn_4BatchNormalization_siv_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr};

  // _bn_4BatchNormalization_closure CLOSURE CALL
  _bn_4BatchNormalization_closure(&DeeployNetwork__bn_4BatchNormalization_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref;
} _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args_t;

static void _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure(
    void *_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args_t *args =
      (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args_t *)
          _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 5, 64] -> [1, 64, 2, 5] (Name: bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 64;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const float32_t *src =
      (const float32_t *)DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref;
  float32_t *dst = (float32_t *)DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_2;
  const uint32_t sOut_1 = sOut_2 * dimLen_1;
  const uint32_t sOut_0 = sOut_1 * dimLen_3;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    const uint32_t baseChunk_1 = dimLen_3 / NUM_CORES;
    const uint32_t leftover_1 = dimLen_3 - baseChunk_1 * NUM_CORES;
    const uint32_t offset_1 = baseChunk_1 * coreId + (coreId < leftover_1 ? coreId : leftover_1);
    const uint32_t chunk_1 = coreId < leftover_1 ? baseChunk_1 + 1 : baseChunk_1;
    for (uint32_t i_3 = offset_1; i_3 < offset_1 + chunk_1; i_3++) {
      const uint32_t offS_1 = offS_0 + i_3 * sIn_3;
      const uint32_t offD_1 = offD_0 + i_3 * sOut_1;

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
        const uint32_t offS_2 = offS_1 + i_1 * sIn_1;
        const uint32_t offD_2 = offD_1 + i_1 * sOut_2;

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
          const uint32_t offS_3 = offS_2 + i_2 * sIn_2;
          const uint32_t offD_3 = offD_2 + i_2 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref;
} _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t;

static void _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork(
    void *_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t *args =
      (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t *)
          _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args_t
      DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args =
          (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref};

  // _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure CLOSURE CALL
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure(
      &DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr;
} _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args_t;

static void _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure(
    void *_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args_t *args =
      (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args_t *)
          _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args;

  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor = args->DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr]);
  void
      *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_bn_4BatchNormalization_output_0_tensor_ref =
          (void *)((char *)DeeployNetwork_bn_4BatchNormalization_output_0_tensor +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1444352,
          .size = 2560,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t
        DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args =
            (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork,
                    &DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1313280,
          .size = 2560,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_bn_4BatchNormalization_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr;
} _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args_t;

static void _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3(
    void *_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args_t *args =
      (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args_t *)
          _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args;

  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor = args->DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args_t
      DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args =
          (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args_t){
              .DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
                  DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed,
              .DeeployNetwork_bn_4BatchNormalization_output_0_tensor = DeeployNetwork_bn_4BatchNormalization_output_0_tensor,
              .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr};

  // _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure CLOSURE CALL
  _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure(
      &DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref;
} _relu_4Relu_tiling_closure_args_t;

static void _relu_4Relu_tiling_closure(void *_relu_4Relu_tiling_closure_args) {
  // CLOSURE ARG CAST
  _relu_4Relu_tiling_closure_args_t *args = (_relu_4Relu_tiling_closure_args_t *)_relu_4Relu_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref;

  // CLOSURE FUNCTION CALL

  // ReLU (Name: relu_4Relu, Op: Relu)
  PULP_Relu_fp32_fp32(DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref, 640);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref;
} _relu_4Relu_cluster_fork_args_t;

static void _relu_4Relu_cluster_fork(void *_relu_4Relu_cluster_fork_args) {
  // CLOSURE ARG CAST
  _relu_4Relu_cluster_fork_args_t *args = (_relu_4Relu_cluster_fork_args_t *)_relu_4Relu_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref;

  // CLOSURE FUNCTION CALL
  _relu_4Relu_tiling_closure_args_t DeeployNetwork__relu_4Relu_tiling_closure_args =
      (_relu_4Relu_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref};

  // _relu_4Relu_tiling_closure CLOSURE CALL
  _relu_4Relu_tiling_closure(&DeeployNetwork__relu_4Relu_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr;
} _relu_4Relu_closure_args_t;

static void _relu_4Relu_closure(void *_relu_4Relu_closure_args) {
  // CLOSURE ARG CAST
  _relu_4Relu_closure_args_t *args = (_relu_4Relu_closure_args_t *)_relu_4Relu_closure_args;

  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor = args->DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor = args->DeeployNetwork_relu_4Relu_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  void *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_bn_4BatchNormalization_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_bn_4BatchNormalization_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_relu_4Relu_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_relu_4Relu_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr]);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1444352,
                                      .size = 2560,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_bn_4BatchNormalization_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _relu_4Relu_cluster_fork_args_t DeeployNetwork__relu_4Relu_cluster_fork_args =
        (_relu_4Relu_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_relu_4Relu_cluster_fork, &DeeployNetwork__relu_4Relu_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1313280,
                                      .size = 2560,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_relu_4Relu_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr;
} _relu_4Relu_closure_L3_args_t;

static void _relu_4Relu_closure_L3(void *_relu_4Relu_closure_L3_args) {
  // CLOSURE ARG CAST
  _relu_4Relu_closure_L3_args_t *args = (_relu_4Relu_closure_L3_args_t *)_relu_4Relu_closure_L3_args;

  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor = args->DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor = args->DeeployNetwork_relu_4Relu_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _relu_4Relu_closure_args_t DeeployNetwork__relu_4Relu_closure_args =
      (_relu_4Relu_closure_args_t){.DeeployNetwork_bn_4BatchNormalization_output_0_tensor = DeeployNetwork_bn_4BatchNormalization_output_0_tensor,
                                   .DeeployNetwork_relu_4Relu_output_0_tensor = DeeployNetwork_relu_4Relu_output_0_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr};

  // _relu_4Relu_closure CLOSURE CALL
  _relu_4Relu_closure(&DeeployNetwork__relu_4Relu_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref;
} _global_poolGlobalAveragePool_tiling_closure_args_t;

static void _global_poolGlobalAveragePool_tiling_closure(void *_global_poolGlobalAveragePool_tiling_closure_args) {
  // CLOSURE ARG CAST
  _global_poolGlobalAveragePool_tiling_closure_args_t *args =
      (_global_poolGlobalAveragePool_tiling_closure_args_t *)_global_poolGlobalAveragePool_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref;

  // CLOSURE FUNCTION CALL

  // GlobalAveragePool (Name: global_poolGlobalAveragePool, Op: GlobalAveragePool)
  PULP_GlobalAveragePool_fp32(DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                              DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref, 1, 64, 2, 5);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref;
} _global_poolGlobalAveragePool_cluster_fork_args_t;

static void _global_poolGlobalAveragePool_cluster_fork(void *_global_poolGlobalAveragePool_cluster_fork_args) {
  // CLOSURE ARG CAST
  _global_poolGlobalAveragePool_cluster_fork_args_t *args =
      (_global_poolGlobalAveragePool_cluster_fork_args_t *)_global_poolGlobalAveragePool_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref;

  // CLOSURE FUNCTION CALL
  _global_poolGlobalAveragePool_tiling_closure_args_t DeeployNetwork__global_poolGlobalAveragePool_tiling_closure_args =
      (_global_poolGlobalAveragePool_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref};

  // _global_poolGlobalAveragePool_tiling_closure CLOSURE CALL
  _global_poolGlobalAveragePool_tiling_closure(&DeeployNetwork__global_poolGlobalAveragePool_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr;
} _global_poolGlobalAveragePool_closure_args_t;

static void _global_poolGlobalAveragePool_closure(void *_global_poolGlobalAveragePool_closure_args) {
  // CLOSURE ARG CAST
  _global_poolGlobalAveragePool_closure_args_t *args = (_global_poolGlobalAveragePool_closure_args_t *)_global_poolGlobalAveragePool_closure_args;

  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor = args->DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor = args->DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  void *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_relu_4Relu_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_relu_4Relu_output_0_tensor + DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_cumByteOffset
                                                                       [*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_global_poolGlobalAveragePool_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr]);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1444352,
                                      .size = 2560,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_relu_4Relu_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _global_poolGlobalAveragePool_cluster_fork_args_t DeeployNetwork__global_poolGlobalAveragePool_cluster_fork_args =
        (_global_poolGlobalAveragePool_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_global_poolGlobalAveragePool_cluster_fork, &DeeployNetwork__global_poolGlobalAveragePool_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310976,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_global_poolGlobalAveragePool_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr;
} _global_poolGlobalAveragePool_closure_L3_args_t;

static void _global_poolGlobalAveragePool_closure_L3(void *_global_poolGlobalAveragePool_closure_L3_args) {
  // CLOSURE ARG CAST
  _global_poolGlobalAveragePool_closure_L3_args_t *args = (_global_poolGlobalAveragePool_closure_L3_args_t *)_global_poolGlobalAveragePool_closure_L3_args;

  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor = args->DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor = args->DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _global_poolGlobalAveragePool_closure_args_t DeeployNetwork__global_poolGlobalAveragePool_closure_args = (_global_poolGlobalAveragePool_closure_args_t){
      .DeeployNetwork_relu_4Relu_output_0_tensor = DeeployNetwork_relu_4Relu_output_0_tensor,
      .DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor = DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr};

  // _global_poolGlobalAveragePool_closure CLOSURE CALL
  _global_poolGlobalAveragePool_closure(&DeeployNetwork__global_poolGlobalAveragePool_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref;

  // CLOSURE FUNCTION CALL

  // DualHalf (Name: wrappedInnerForwardImpl_5Gemm_half0__dual65, Op: DualHalf, half 0) -- QW exp8
  // contiguous per-core chunk: the strided form (c = core_id; c += NUM_CORES) lost the last iteration of cores 1..7 on
  // Siracusa (LLVM hardware-loop trip count), GAP9 was fine
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual65_chunk = (32u + NUM_CORES - 1u) / NUM_CORES;
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual65_start = MIN(wrappedInnerForwardImpl_5Gemm_half0__dual65_chunk * (uint32_t)pi_core_id(), 32u);
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual65_stop =
      MIN(wrappedInnerForwardImpl_5Gemm_half0__dual65_start + wrappedInnerForwardImpl_5Gemm_half0__dual65_chunk, 32u);
  for (uint32_t r = 0; r < 1u; r++) {
    for (uint32_t c = wrappedInnerForwardImpl_5Gemm_half0__dual65_start; c < wrappedInnerForwardImpl_5Gemm_half0__dual65_stop; c++)
      ((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref)[r * 32u + c] =
          ((const float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref)[r * 2u * 32u + 0u * 32u + c];
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure(void *_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor = args->DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_Reshape_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_Reshape_output_0_p__dual64_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_p__dual64_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_Reshape_output_0_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_Reshape_output_0_p__dual64_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor = args->DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args_t){
          .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
          .DeeployNetwork_Reshape_output_0_p__dual64_tensor = DeeployNetwork_Reshape_output_0_p__dual64_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: wrappedInnerForwardImpl_5Gemm, Op: Gemm)
  float32_t
      *ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref;
  float32_t
      *ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref;
  float32_t
      *ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref;

  for (uint32_t i = 0; i < 1; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref,
        1, 32, 9, 0, 1, 0);
    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref +=
        1 * 32;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref +=
        32 * 9;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref +=
        1 * 9;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure(void *_wrappedInnerForwardImpl_5Gemm_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor = args->DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1316);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_p__dual64_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_p__dual64_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_p__dual68_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset
                                                                                [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_p__dual70_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_output_tensor_ref =
      (void *)((char *)DeeployNetwork_output_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset
                                                          [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);

  // Initialize DMA futures
  int B_ExternalToLocal = -1;
  int A_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int C_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    // Transfer input tiles
    A_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_p__dual64_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    B_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_p__dual68_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    C_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_p__dual70_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (B_ExternalToLocal >= 0) {
      mchan_transfer_wait(B_ExternalToLocal);
      mchan_transfer_free(B_ExternalToLocal);
    }

    if (A_ExternalToLocal >= 0) {
      mchan_transfer_wait(A_ExternalToLocal);
      mchan_transfer_free(A_ExternalToLocal);
    }

    if (C_ExternalToLocal >= 0) {
      mchan_transfer_wait(C_ExternalToLocal);
      mchan_transfer_free(C_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_cluster_fork, &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_output_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor = args->DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor,
      .DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor,
      .DeeployNetwork_Reshape_output_0_p__dual64_tensor = DeeployNetwork_Reshape_output_0_p__dual64_tensor,
      .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref;

  // CLOSURE FUNCTION CALL

  // DualHalf (Name: wrappedInnerForwardImpl_5Gemm_half1__dual67, Op: DualHalf, half 1) -- QW exp8
  // contiguous per-core chunk: the strided form (c = core_id; c += NUM_CORES) lost the last iteration of cores 1..7 on
  // Siracusa (LLVM hardware-loop trip count), GAP9 was fine
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual67_chunk = (32u + NUM_CORES - 1u) / NUM_CORES;
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual67_start = MIN(wrappedInnerForwardImpl_5Gemm_half1__dual67_chunk * (uint32_t)pi_core_id(), 32u);
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual67_stop =
      MIN(wrappedInnerForwardImpl_5Gemm_half1__dual67_start + wrappedInnerForwardImpl_5Gemm_half1__dual67_chunk, 32u);
  for (uint32_t r = 0; r < 1u; r++) {
    for (uint32_t c = wrappedInnerForwardImpl_5Gemm_half1__dual67_start; c < wrappedInnerForwardImpl_5Gemm_half1__dual67_stop; c++)
      ((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref)[r * 32u + c] =
          ((const float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref)[r * 2u * 32u + 1u * 32u + c];
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure(void *_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor = args->DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_Reshape_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_Reshape_output_0_m__dual66_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_m__dual66_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_Reshape_output_0_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_Reshape_output_0_m__dual66_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor = args->DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args_t){
          .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
          .DeeployNetwork_Reshape_output_0_m__dual66_tensor = DeeployNetwork_Reshape_output_0_m__dual66_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: wrappedInnerForwardImpl_5Gemm_minus__dual73, Op: Gemm)
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref;

  for (uint32_t i = 0; i < 1; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref,
        1, 32, 9, 0, 1, 0);
    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref +=
        1 * 32;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref +=
        32 * 9;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref +=
        1 * 9;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  float32_t *DeeployNetwork_output_minus__dual72_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure(void *_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor = args->DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  float32_t *DeeployNetwork_output_minus__dual72_tensor = args->DeeployNetwork_output_minus__dual72_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1316);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_Reshape_output_0_m__dual66_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_m__dual66_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_fcweight_fp32_pert_m__dual74_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_fcbias_fp32_pert_m__dual76_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_output_minus__dual72_tensor_ref =
      (void *)((char *)DeeployNetwork_output_minus__dual72_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr]);

  // Initialize DMA futures
  int B_ExternalToLocal = -1;
  int A_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int C_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    A_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_Reshape_output_0_m__dual66_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    B_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_fcweight_fp32_pert_m__dual74_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    C_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_fcbias_fp32_pert_m__dual76_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (B_ExternalToLocal >= 0) {
      mchan_transfer_wait(B_ExternalToLocal);
      mchan_transfer_free(B_ExternalToLocal);
    }

    if (A_ExternalToLocal >= 0) {
      mchan_transfer_wait(A_ExternalToLocal);
      mchan_transfer_free(A_ExternalToLocal);
    }

    if (C_ExternalToLocal >= 0) {
      mchan_transfer_wait(C_ExternalToLocal);
      mchan_transfer_free(C_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_A_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_B_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_C_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_output_minus__dual72_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  float32_t *DeeployNetwork_output_minus__dual72_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor = args->DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  float32_t *DeeployNetwork_output_minus__dual72_tensor = args->DeeployNetwork_output_minus__dual72_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args_t){
          .DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor,
          .DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor,
          .DeeployNetwork_Reshape_output_0_m__dual66_tensor = DeeployNetwork_Reshape_output_0_m__dual66_tensor,
          .DeeployNetwork_output_minus__dual72_tensor = DeeployNetwork_output_minus__dual72_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref;
} _CrossEntropyLoss_tiling_closure_args_t;

static void _CrossEntropyLoss_tiling_closure(void *_CrossEntropyLoss_tiling_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_tiling_closure_args_t *args = (_CrossEntropyLoss_tiling_closure_args_t *)_CrossEntropyLoss_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref;

  // CLOSURE FUNCTION CALL

  BEGIN_SINGLE_CORE
  // SoftmaxCrossEntropyLoss (Name: CrossEntropyLoss, Op: SoftmaxCrossEntropyLoss)
  float32_t sce_total_loss = 0.0f;
  for (uint32_t i = 0; i < 1; i++) {
    float32_t sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9];
    for (uint32_t j = 1; j < 9; j++) {
      if (DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9 + j] > sce_max_logit)
        sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9 + j];
    }
    float32_t sce_sum_exp = 0.0f;
    for (uint32_t j = 0; j < 9; j++)
      sce_sum_exp += expf(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9 + j] - sce_max_logit);
    float32_t sce_log_sum_exp = logf(sce_sum_exp);
    for (uint32_t j = 0; j < 9; j++)
      DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref[i * 9 + j] =
          DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9 + j] - sce_max_logit - sce_log_sum_exp;
    sce_total_loss +=
        -(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref[i * 9 + (uint32_t)(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref[i])] -
          sce_max_logit - sce_log_sum_exp);
  }
  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref[0] = sce_total_loss / (float32_t)1;
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref;
} _CrossEntropyLoss_cluster_fork_args_t;

static void _CrossEntropyLoss_cluster_fork(void *_CrossEntropyLoss_cluster_fork_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_cluster_fork_args_t *args = (_CrossEntropyLoss_cluster_fork_args_t *)_CrossEntropyLoss_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_tiling_closure_args_t DeeployNetwork__CrossEntropyLoss_tiling_closure_args = (_CrossEntropyLoss_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref};

  // _CrossEntropyLoss_tiling_closure CLOSURE CALL
  _CrossEntropyLoss_tiling_closure(&DeeployNetwork__CrossEntropyLoss_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_log_prob_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
} _CrossEntropyLoss_closure_args_t;

static void _CrossEntropyLoss_closure(void *_CrossEntropyLoss_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_closure_args_t *args = (_CrossEntropyLoss_closure_args_t *)_CrossEntropyLoss_closure_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_log_prob_tensor = args->DeeployNetwork_log_prob_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12);
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 48);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_tensor_ref =
      (void *)((char *)DeeployNetwork_output_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_tensor_ref =
      (void *)((char *)DeeployNetwork_log_prob_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);

  // Initialize DMA futures
  int loss_LocalToExternal = -1;
  int log_prob_LocalToExternal = -1;
  int labels_ExternalToLocal = -1;
  int logits_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    logits_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    labels_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441800,
                                      .size = 8,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (labels_ExternalToLocal >= 0) {
      mchan_transfer_wait(labels_ExternalToLocal);
      mchan_transfer_free(labels_ExternalToLocal);
    }

    if (logits_ExternalToLocal >= 0) {
      mchan_transfer_wait(logits_ExternalToLocal);
      mchan_transfer_free(logits_ExternalToLocal);
    }

    _CrossEntropyLoss_cluster_fork_args_t DeeployNetwork__CrossEntropyLoss_cluster_fork_args = (_CrossEntropyLoss_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_CrossEntropyLoss_cluster_fork, &DeeployNetwork__CrossEntropyLoss_cluster_fork_args);

    // Transfer output tiles
    log_prob_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    loss_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310724,
                                      .size = 4,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (log_prob_LocalToExternal >= 0) {
      mchan_transfer_wait(log_prob_LocalToExternal);
      mchan_transfer_free(log_prob_LocalToExternal);
    }

    if (loss_LocalToExternal >= 0) {
      mchan_transfer_wait(loss_LocalToExternal);
      mchan_transfer_free(loss_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_log_prob_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
} _CrossEntropyLoss_closure_L3_args_t;

static void _CrossEntropyLoss_closure_L3(void *_CrossEntropyLoss_closure_L3_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_closure_L3_args_t *args = (_CrossEntropyLoss_closure_L3_args_t *)_CrossEntropyLoss_closure_L3_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_log_prob_tensor = args->DeeployNetwork_log_prob_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_closure_args_t DeeployNetwork__CrossEntropyLoss_closure_args = (_CrossEntropyLoss_closure_args_t){
      .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
      .DeeployNetwork_log_prob_tensor = DeeployNetwork_log_prob_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr};

  // _CrossEntropyLoss_closure CLOSURE CALL
  _CrossEntropyLoss_closure(&DeeployNetwork__CrossEntropyLoss_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref;
} _CrossEntropyLoss_minus__dual79_tiling_closure_args_t;

static void _CrossEntropyLoss_minus__dual79_tiling_closure(void *_CrossEntropyLoss_minus__dual79_tiling_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual79_tiling_closure_args_t *args =
      (_CrossEntropyLoss_minus__dual79_tiling_closure_args_t *)_CrossEntropyLoss_minus__dual79_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref;

  // CLOSURE FUNCTION CALL

  BEGIN_SINGLE_CORE
  // SoftmaxCrossEntropyLoss (Name: CrossEntropyLoss_minus__dual79, Op: SoftmaxCrossEntropyLoss)
  float32_t sce_total_loss = 0.0f;
  for (uint32_t i = 0; i < 1; i++) {
    float32_t sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref[i * 9];
    for (uint32_t j = 1; j < 9; j++) {
      if (DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref[i * 9 + j] > sce_max_logit)
        sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref[i * 9 + j];
    }
    float32_t sce_sum_exp = 0.0f;
    for (uint32_t j = 0; j < 9; j++)
      sce_sum_exp += expf(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref[i * 9 + j] - sce_max_logit);
    float32_t sce_log_sum_exp = logf(sce_sum_exp);
    for (uint32_t j = 0; j < 9; j++)
      DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref[i * 9 + j] =
          DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref[i * 9 + j] - sce_max_logit - sce_log_sum_exp;
    sce_total_loss += -(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref
                            [i * 9 + (uint32_t)(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref[i])] -
                        sce_max_logit - sce_log_sum_exp);
  }
  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref[0] = sce_total_loss / (float32_t)1;
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref;
} _CrossEntropyLoss_minus__dual79_cluster_fork_args_t;

static void _CrossEntropyLoss_minus__dual79_cluster_fork(void *_CrossEntropyLoss_minus__dual79_cluster_fork_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual79_cluster_fork_args_t *args =
      (_CrossEntropyLoss_minus__dual79_cluster_fork_args_t *)_CrossEntropyLoss_minus__dual79_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_minus__dual79_tiling_closure_args_t DeeployNetwork__CrossEntropyLoss_minus__dual79_tiling_closure_args =
      (_CrossEntropyLoss_minus__dual79_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref};

  // _CrossEntropyLoss_minus__dual79_tiling_closure CLOSURE CALL
  _CrossEntropyLoss_minus__dual79_tiling_closure(&DeeployNetwork__CrossEntropyLoss_minus__dual79_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_minus__dual72_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual78_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr;
} _CrossEntropyLoss_minus__dual79_closure_args_t;

static void _CrossEntropyLoss_minus__dual79_closure(void *_CrossEntropyLoss_minus__dual79_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual79_closure_args_t *args = (_CrossEntropyLoss_minus__dual79_closure_args_t *)_CrossEntropyLoss_minus__dual79_closure_args;

  float32_t *DeeployNetwork_output_minus__dual72_tensor = args->DeeployNetwork_output_minus__dual72_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual78_tensor = args->DeeployNetwork_log_prob_minus__dual78_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12);
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 48);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_output_minus__dual72_tensor_ref =
      (void *)((char *)DeeployNetwork_output_minus__dual72_tensor + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_minus__dual78_tensor_ref =
      (void *)((char *)DeeployNetwork_log_prob_minus__dual78_tensor + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_cumByteOffset
                                                                          [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_output_1_ref =
      (void *)((char *)DeeployNetwork_output_1 + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr]);

  // Initialize DMA futures
  int loss_LocalToExternal = -1;
  int log_prob_LocalToExternal = -1;
  int labels_ExternalToLocal = -1;
  int logits_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    logits_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_output_minus__dual72_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    labels_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441800,
                                      .size = 8,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (labels_ExternalToLocal >= 0) {
      mchan_transfer_wait(labels_ExternalToLocal);
      mchan_transfer_free(labels_ExternalToLocal);
    }

    if (logits_ExternalToLocal >= 0) {
      mchan_transfer_wait(logits_ExternalToLocal);
      mchan_transfer_free(logits_ExternalToLocal);
    }

    _CrossEntropyLoss_minus__dual79_cluster_fork_args_t DeeployNetwork__CrossEntropyLoss_minus__dual79_cluster_fork_args =
        (_CrossEntropyLoss_minus__dual79_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_logits_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_labels_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_CrossEntropyLoss_minus__dual79_cluster_fork, &DeeployNetwork__CrossEntropyLoss_minus__dual79_cluster_fork_args);

    // Transfer output tiles
    log_prob_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_log_prob_minus__dual78_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    loss_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310724,
                                      .size = 4,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_loss_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_output_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (log_prob_LocalToExternal >= 0) {
      mchan_transfer_wait(log_prob_LocalToExternal);
      mchan_transfer_free(log_prob_LocalToExternal);
    }

    if (loss_LocalToExternal >= 0) {
      mchan_transfer_wait(loss_LocalToExternal);
      mchan_transfer_free(loss_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_minus__dual72_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual78_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr;
} _CrossEntropyLoss_minus__dual79_closure_L3_args_t;

static void _CrossEntropyLoss_minus__dual79_closure_L3(void *_CrossEntropyLoss_minus__dual79_closure_L3_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual79_closure_L3_args_t *args =
      (_CrossEntropyLoss_minus__dual79_closure_L3_args_t *)_CrossEntropyLoss_minus__dual79_closure_L3_args;

  float32_t *DeeployNetwork_output_minus__dual72_tensor = args->DeeployNetwork_output_minus__dual72_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual78_tensor = args->DeeployNetwork_log_prob_minus__dual78_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_minus__dual79_closure_args_t DeeployNetwork__CrossEntropyLoss_minus__dual79_closure_args = (_CrossEntropyLoss_minus__dual79_closure_args_t){
      .DeeployNetwork_output_minus__dual72_tensor = DeeployNetwork_output_minus__dual72_tensor,
      .DeeployNetwork_log_prob_minus__dual78_tensor = DeeployNetwork_log_prob_minus__dual78_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr};

  // _CrossEntropyLoss_minus__dual79_closure CLOSURE CALL
  _CrossEntropyLoss_minus__dual79_closure(&DeeployNetwork__CrossEntropyLoss_minus__dual79_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor;
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor;
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor;
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor;
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor;
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor;
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor;
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor;
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor;
  uint8_t *DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor;
  uint8_t *DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor;
  uint8_t *DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  uint8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  int8_t *DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor;
  int8_t *DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor;
  int8_t *DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual64_tensor;
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual66_tensor;
  float32_t *DeeployNetwork_output_minus__dual72_tensor;
  float32_t *DeeployNetwork_log_prob_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual78_tensor;
  void *DeeployNetwork_pert_fcbias_fp32_mode1__dual71_scratch;
  void *DeeployNetwork_pert_fcbias_fp32_mode2__dual77_scratch;
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16bias__dual4_buffer;
  void *DeeployNetwork_pert_blocks0bnweight_mode0__dual14_scratch;
  void *DeeployNetwork_pert_blocks0bnbias_mode0__dual16_scratch;
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_buffer;
  void *DeeployNetwork_pert_blocks1bnweight_mode0__dual29_scratch;
  void *DeeployNetwork_pert_blocks1bnbias_mode0__dual31_scratch;
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_buffer;
  void *DeeployNetwork_pert_blocks2bnweight_mode0__dual44_scratch;
  void *DeeployNetwork_pert_blocks2bnbias_mode0__dual46_scratch;
  void *DeeployNetwork_pert_blocks3bnweight_mode0__dual51_scratch;
  void *DeeployNetwork_pert_blocks3bnbias_mode0__dual53_scratch;
  void *DeeployNetwork_pert_blocks4bnweight_mode0__dual58_scratch;
  void *DeeployNetwork_pert_blocks4bnbias_mode0__dual60_scratch;
  void *DeeployNetwork_wrappedInnerForwardImplConv_ne16wenc__dual3_buffer;
  void *DeeployNetwork_pert_fcweight_fp32_mode1__dual69_scratch;
  void *DeeployNetwork_pert_fcweight_fp32_mode2__dual75_scratch;
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_ne16__dual26_colbuf;
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv_ne16__dual41_colbuf;
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr;
    DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    _pert_fcbias_fp32_mode1__dual71_closure_L3_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual71_closure_L3_args =
        (_pert_fcbias_fp32_mode1__dual71_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor =
                                                                DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual71_tileIdxPtr};

    // _pert_fcbias_fp32_mode1__dual71_closure_L3 CLOSURE CALL
    _pert_fcbias_fp32_mode1__dual71_closure_L3(&DeeployNetwork__pert_fcbias_fp32_mode1__dual71_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr;
    DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56524);
    _pert_fcbias_fp32_mode2__dual77_closure_L3_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual77_closure_L3_args =
        (_pert_fcbias_fp32_mode2__dual77_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor =
                                                                DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual77_tileIdxPtr};

    // _pert_fcbias_fp32_mode2__dual77_closure_L3 CLOSURE CALL
    _pert_fcbias_fp32_mode2__dual77_closure_L3(&DeeployNetwork__pert_fcbias_fp32_mode2__dual77_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr;
    DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58964);
    _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args =
        (_wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args_t){
            .DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16bias__dual4_tileIdxPtr};

    // _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3(&DeeployNetwork__wrappedInnerForwardImplConv_ne16bias__dual4_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr;
    DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _pert_blocks0bnweight_mode0__dual14_closure_L3_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual14_closure_L3_args =
        (_pert_blocks0bnweight_mode0__dual14_closure_L3_args_t){.DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor =
                                                                    DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual14_tileIdxPtr};

    // _pert_blocks0bnweight_mode0__dual14_closure_L3 CLOSURE CALL
    _pert_blocks0bnweight_mode0__dual14_closure_L3(&DeeployNetwork__pert_blocks0bnweight_mode0__dual14_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr;
    DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59028);
    _pert_blocks0bnbias_mode0__dual16_closure_L3_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual16_closure_L3_args =
        (_pert_blocks0bnbias_mode0__dual16_closure_L3_args_t){.DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor =
                                                                  DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual16_tileIdxPtr};

    // _pert_blocks0bnbias_mode0__dual16_closure_L3 CLOSURE CALL
    _pert_blocks0bnbias_mode0__dual16_closure_L3(&DeeployNetwork__pert_blocks0bnbias_mode0__dual16_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr;
    DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216372);
    _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args =
        (_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args_t){
            .DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16bias__dual22_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16bias__dual22_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr;
    DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216500);
    _pert_blocks1bnweight_mode0__dual29_closure_L3_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual29_closure_L3_args =
        (_pert_blocks1bnweight_mode0__dual29_closure_L3_args_t){.DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor =
                                                                    DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual29_tileIdxPtr};

    // _pert_blocks1bnweight_mode0__dual29_closure_L3 CLOSURE CALL
    _pert_blocks1bnweight_mode0__dual29_closure_L3(&DeeployNetwork__pert_blocks1bnweight_mode0__dual29_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr;
    DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217140);
    _pert_blocks1bnbias_mode0__dual31_closure_L3_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual31_closure_L3_args =
        (_pert_blocks1bnbias_mode0__dual31_closure_L3_args_t){.DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor =
                                                                  DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual31_tileIdxPtr};

    // _pert_blocks1bnbias_mode0__dual31_closure_L3 CLOSURE CALL
    _pert_blocks1bnbias_mode0__dual31_closure_L3(&DeeployNetwork__pert_blocks1bnbias_mode0__dual31_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr;
    DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217268);
    _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args =
        (_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args_t){
            .DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16bias__dual37_tileIdxPtr};

    // _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16bias__dual37_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr;
    DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217396);
    _pert_blocks2bnweight_mode0__dual44_closure_L3_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual44_closure_L3_args =
        (_pert_blocks2bnweight_mode0__dual44_closure_L3_args_t){.DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor =
                                                                    DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual44_tileIdxPtr};

    // _pert_blocks2bnweight_mode0__dual44_closure_L3 CLOSURE CALL
    _pert_blocks2bnweight_mode0__dual44_closure_L3(&DeeployNetwork__pert_blocks2bnweight_mode0__dual44_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr;
    DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 218036);
    _pert_blocks2bnbias_mode0__dual46_closure_L3_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual46_closure_L3_args =
        (_pert_blocks2bnbias_mode0__dual46_closure_L3_args_t){.DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor =
                                                                  DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual46_tileIdxPtr};

    // _pert_blocks2bnbias_mode0__dual46_closure_L3 CLOSURE CALL
    _pert_blocks2bnbias_mode0__dual46_closure_L3(&DeeployNetwork__pert_blocks2bnbias_mode0__dual46_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr;
    DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216628);
    _pert_blocks3bnweight_mode0__dual51_closure_L3_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual51_closure_L3_args =
        (_pert_blocks3bnweight_mode0__dual51_closure_L3_args_t){.DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor =
                                                                    DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual51_tileIdxPtr};

    // _pert_blocks3bnweight_mode0__dual51_closure_L3 CLOSURE CALL
    _pert_blocks3bnweight_mode0__dual51_closure_L3(&DeeployNetwork__pert_blocks3bnweight_mode0__dual51_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr;
    DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217524);
    _pert_blocks3bnbias_mode0__dual53_closure_L3_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual53_closure_L3_args =
        (_pert_blocks3bnbias_mode0__dual53_closure_L3_args_t){.DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor =
                                                                  DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual53_tileIdxPtr};

    // _pert_blocks3bnbias_mode0__dual53_closure_L3 CLOSURE CALL
    _pert_blocks3bnbias_mode0__dual53_closure_L3(&DeeployNetwork__pert_blocks3bnbias_mode0__dual53_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr;
    DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216884);
    _pert_blocks4bnweight_mode0__dual58_closure_L3_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual58_closure_L3_args =
        (_pert_blocks4bnweight_mode0__dual58_closure_L3_args_t){.DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor =
                                                                    DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual58_tileIdxPtr};

    // _pert_blocks4bnweight_mode0__dual58_closure_L3 CLOSURE CALL
    _pert_blocks4bnweight_mode0__dual58_closure_L3(&DeeployNetwork__pert_blocks4bnweight_mode0__dual58_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr;
    DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217780);
    _pert_blocks4bnbias_mode0__dual60_closure_L3_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual60_closure_L3_args =
        (_pert_blocks4bnbias_mode0__dual60_closure_L3_args_t){.DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor =
                                                                  DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual60_tileIdxPtr};

    // _pert_blocks4bnbias_mode0__dual60_closure_L3 CLOSURE CALL
    _pert_blocks4bnbias_mode0__dual60_closure_L3(&DeeployNetwork__pert_blocks4bnbias_mode0__dual60_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr;
    DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59092);
    _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args =
        (_wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args_t){
            .DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_ne16wenc__dual3_tileIdxPtr};

    // _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3(&DeeployNetwork__wrappedInnerForwardImplConv_ne16wenc__dual3_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr;
    DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56596);
    _pert_fcweight_fp32_mode1__dual69_closure_L3_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual69_closure_L3_args =
        (_pert_fcweight_fp32_mode1__dual69_closure_L3_args_t){.DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor =
                                                                  DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual69_tileIdxPtr};

    // _pert_fcweight_fp32_mode1__dual69_closure_L3 CLOSURE CALL
    _pert_fcweight_fp32_mode1__dual69_closure_L3(&DeeployNetwork__pert_fcweight_fp32_mode1__dual69_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr;
    DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 57748);
    _pert_fcweight_fp32_mode2__dual75_closure_L3_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual75_closure_L3_args =
        (_pert_fcweight_fp32_mode2__dual75_closure_L3_args_t){.DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor =
                                                                  DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual75_tileIdxPtr};

    // _pert_fcweight_fp32_mode2__dual75_closure_L3 CLOSURE CALL
    _pert_fcweight_fp32_mode2__dual75_closure_L3(&DeeployNetwork__pert_fcweight_fp32_mode2__dual75_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr;
    DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 222260);
    _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args =
        (_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args_t){
            .DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16wenc__dual21_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr;
    DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 218164);
    _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args =
        (_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args_t){
            .DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_tileIdxPtr};

    // _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16wenc__dual36_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;
    DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59348);
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args =
        (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t){
            .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr};

    // _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3 CLOSURE CALL
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3(&DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr;
    DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 226356);
    _wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args =
        (_wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args_t){
            .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
            .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor =
                DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv_im2col__dual7_tileIdxPtr};

    // _wrappedInnerForwardImplConv_im2col__dual7_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv_im2col__dual7_closure_L3(&DeeployNetwork__wrappedInnerForwardImplConv_im2col__dual7_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshape, Op: Reshape)
    DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped =
        DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59348);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor = DeeployNetwork_blocks0convbias_rqsadd_ne16add2__dual2_tensor,
        .DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor = DeeployNetwork_blocks0convweight_int8_ne16enc__dual1_tensor,
        .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped =
            DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_input_quant_1Clip_output_0_col__dual6_tensor_Reshaped,
        .DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped =
            DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped_Reshape, Op: Reshape)
    DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor =
        DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor_Reshaped;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 226356);
    _bnBatchNormalization_tail_closure_L3_args_t DeeployNetwork__bnBatchNormalization_tail_closure_L3_args = (_bnBatchNormalization_tail_closure_L3_args_t){
        .DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor = DeeployNetwork_blocks0bnweight_pert_pm__dual13_tensor,
        .DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor = DeeployNetwork_blocks0bnbias_pert_pm__dual15_tensor,
        .DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor =
            DeeployNetwork_blocks_0_conv_output_quant_1Clip_output_0_hwc__dual5_tensor,
        .DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
            DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

    // _bnBatchNormalization_tail_closure_L3 CLOSURE CALL
    _bnBatchNormalization_tail_closure_L3(&DeeployNetwork__bnBatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr;
    DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args =
        (_wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args_t){
            .DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor = DeeployNetwork_blocks1convbias_rqsadd_ne16add2__dual20_tensor,
            .DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor = DeeployNetwork_blocks1convweight_int8_ne16enc__dual19_tensor,
            .DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor =
                DeeployNetwork_blocks_1_conv_input_quant_1Clip_output_0_hwc__dual24_tensor,
            .DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
                DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_ne16__dual26_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_ne16__dual26_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 98324);
    _bn_1BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args =
        (_bn_1BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor = DeeployNetwork_blocks1bnweight_pert_pm__dual28_tensor,
            .DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor = DeeployNetwork_blocks1bnbias_pert_pm__dual30_tensor,
            .DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor =
                DeeployNetwork_blocks_1_conv_output_quant_1Clip_output_0_hwc__dual23_tensor,
            .DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
                DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

    // _bn_1BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_1BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr;
    DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args =
        (_wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args_t){
            .DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor = DeeployNetwork_blocks2convbias_rqsadd_ne16add2__dual35_tensor,
            .DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor = DeeployNetwork_blocks2convweight_int8_ne16enc__dual34_tensor,
            .DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor =
                DeeployNetwork_blocks_2_conv_input_quant_1Clip_output_0_hwc__dual39_tensor,
            .DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
                DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv_ne16__dual41_tileIdxPtr};

    // _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_2Conv_ne16__dual41_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 69204);
    _bn_2BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args =
        (_bn_2BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor = DeeployNetwork_blocks2bnweight_pert_pm__dual43_tensor,
            .DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor = DeeployNetwork_blocks2bnbias_pert_pm__dual45_tensor,
            .DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor =
                DeeployNetwork_blocks_2_conv_output_quant_1Clip_output_0_hwc__dual38_tensor,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

    // _bn_2BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_2BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args =
        (_wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual49_tileIdxPtr};

    // _wrappedInnerForwardImpl_3Conv__dual49_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_3Conv__dual49_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual49_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61460);
    _bn_3BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args =
        (_bn_3BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor = DeeployNetwork_blocks3bnweight_pert_pm__dual50_tensor,
            .DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor = DeeployNetwork_blocks3bnbias_pert_pm__dual52_tensor,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual49_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

    // _bn_3BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_3BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 64020);
    _wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args =
        (_wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual56_tileIdxPtr};

    // _wrappedInnerForwardImpl_4Conv__dual56_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_4Conv__dual56_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual56_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
    DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    DeeployNetwork_bn_4BatchNormalization_sm_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 62228);
    DeeployNetwork_bn_4BatchNormalization_siv_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61972);
    _bn_4BatchNormalization_closure_L3_args_t DeeployNetwork__bn_4BatchNormalization_closure_L3_args = (_bn_4BatchNormalization_closure_L3_args_t){
        .DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor = DeeployNetwork_blocks4bnweight_pert_pm__dual57_tensor,
        .DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor = DeeployNetwork_blocks4bnbias_pert_pm__dual59_tensor,
        .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual56_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
            DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed,
        .DeeployNetwork_bn_4BatchNormalization_sm_tensor = DeeployNetwork_bn_4BatchNormalization_sm_tensor,
        .DeeployNetwork_bn_4BatchNormalization_siv_tensor = DeeployNetwork_bn_4BatchNormalization_siv_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr};

    // _bn_4BatchNormalization_closure_L3 CLOSURE CALL
    _bn_4BatchNormalization_closure_L3(&DeeployNetwork__bn_4BatchNormalization_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr;
    DeeployNetwork_bn_4BatchNormalization_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61716);
    _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args_t
        DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args =
            (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args_t){
                .DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
                    DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed,
                .DeeployNetwork_bn_4BatchNormalization_output_0_tensor = DeeployNetwork_bn_4BatchNormalization_output_0_tensor,
                .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr};

    // _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3 CLOSURE CALL
    _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3(
        &DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr;
    DeeployNetwork_relu_4Relu_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59156);
    _relu_4Relu_closure_L3_args_t DeeployNetwork__relu_4Relu_closure_L3_args =
        (_relu_4Relu_closure_L3_args_t){.DeeployNetwork_bn_4BatchNormalization_output_0_tensor = DeeployNetwork_bn_4BatchNormalization_output_0_tensor,
                                        .DeeployNetwork_relu_4Relu_output_0_tensor = DeeployNetwork_relu_4Relu_output_0_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr};

    // _relu_4Relu_closure_L3 CLOSURE CALL
    _relu_4Relu_closure_L3(&DeeployNetwork__relu_4Relu_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr;
    DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _global_poolGlobalAveragePool_closure_L3_args_t DeeployNetwork__global_poolGlobalAveragePool_closure_L3_args =
        (_global_poolGlobalAveragePool_closure_L3_args_t){.DeeployNetwork_relu_4Relu_output_0_tensor = DeeployNetwork_relu_4Relu_output_0_tensor,
                                                          .DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor =
                                                              DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor,
                                                          .DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr =
                                                              DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr};

    // _global_poolGlobalAveragePool_closure_L3 CLOSURE CALL
    _global_poolGlobalAveragePool_closure_L3(&DeeployNetwork__global_poolGlobalAveragePool_closure_L3_args);
  }
  {

    // Reshape (Name: Reshape, Op: Reshape)
    DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr;
    DeeployNetwork_Reshape_output_0_p__dual64_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59192);
    _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args_t){
            .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
            .DeeployNetwork_Reshape_output_0_p__dual64_tensor = DeeployNetwork_Reshape_output_0_p__dual64_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual65_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual65_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
    DeeployNetwork_output_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59156);
    _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual70_tensor,
                                                           .DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor =
                                                               DeeployNetwork_fcweight_fp32_pert_p__dual68_tensor,
                                                           .DeeployNetwork_Reshape_output_0_p__dual64_tensor = DeeployNetwork_Reshape_output_0_p__dual64_tensor,
                                                           .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr;
    DeeployNetwork_Reshape_output_0_m__dual66_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args_t){
            .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
            .DeeployNetwork_Reshape_output_0_m__dual66_tensor = DeeployNetwork_Reshape_output_0_m__dual66_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual67_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual67_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr;
    DeeployNetwork_output_minus__dual72_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args_t){
            .DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual76_tensor,
            .DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor = DeeployNetwork_fcweight_fp32_pert_m__dual74_tensor,
            .DeeployNetwork_Reshape_output_0_m__dual66_tensor = DeeployNetwork_Reshape_output_0_m__dual66_tensor,
            .DeeployNetwork_output_minus__dual72_tensor = DeeployNetwork_output_minus__dual72_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual73_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual73_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
    DeeployNetwork_log_prob_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56528);
    _CrossEntropyLoss_closure_L3_args_t DeeployNetwork__CrossEntropyLoss_closure_L3_args = (_CrossEntropyLoss_closure_L3_args_t){
        .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
        .DeeployNetwork_log_prob_tensor = DeeployNetwork_log_prob_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr};

    // _CrossEntropyLoss_closure_L3 CLOSURE CALL
    _CrossEntropyLoss_closure_L3(&DeeployNetwork__CrossEntropyLoss_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr;
    DeeployNetwork_log_prob_minus__dual78_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56532);
    _CrossEntropyLoss_minus__dual79_closure_L3_args_t DeeployNetwork__CrossEntropyLoss_minus__dual79_closure_L3_args =
        (_CrossEntropyLoss_minus__dual79_closure_L3_args_t){.DeeployNetwork_output_minus__dual72_tensor = DeeployNetwork_output_minus__dual72_tensor,
                                                            .DeeployNetwork_log_prob_minus__dual78_tensor = DeeployNetwork_log_prob_minus__dual78_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual79_tileIdxPtr};

    // _CrossEntropyLoss_minus__dual79_closure_L3 CLOSURE CALL
    _CrossEntropyLoss_minus__dual79_closure_L3(&DeeployNetwork__CrossEntropyLoss_minus__dual79_closure_L3_args);
  }
}

void InitTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 113728);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 113728));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 265612);

  DeeployNetwork_input_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56516);
  DeeployNetwork_input_2 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56452);
  DeeployNetwork_input_3 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56484);
  DeeployNetwork_input_4 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56032);
  DeeployNetwork_input_5 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56096);
  DeeployNetwork_input_6 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56224);
  DeeployNetwork_input_7 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56288);
  DeeployNetwork_input_8 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55200);
  DeeployNetwork_input_9 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55328);
  DeeployNetwork_input_10 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55456);
  DeeployNetwork_input_11 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55584);
  DeeployNetwork_input_12 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56420);
  DeeployNetwork_input_13 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 49952);
  DeeployNetwork_input_14 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 52000);
  DeeployNetwork_input_15 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 46368);
  DeeployNetwork_input_16 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39200);
  DeeployNetwork_input_17 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56388);
  DeeployNetwork_input_18 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55968);
  DeeployNetwork_input_19 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56160);
  DeeployNetwork_input_20 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55712);
  DeeployNetwork_input_21 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55840);
  DeeployNetwork_input_22 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 54048);
  DeeployNetwork_input_23 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56352);
  DeeployNetwork_output_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56524);
  DeeployNetwork_output_1 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56528);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_inputs[4] = (void *)DeeployNetwork_input_4;
  DeeployNetwork_inputs[5] = (void *)DeeployNetwork_input_5;
  DeeployNetwork_inputs[6] = (void *)DeeployNetwork_input_6;
  DeeployNetwork_inputs[7] = (void *)DeeployNetwork_input_7;
  DeeployNetwork_inputs[8] = (void *)DeeployNetwork_input_8;
  DeeployNetwork_inputs[9] = (void *)DeeployNetwork_input_9;
  DeeployNetwork_inputs[10] = (void *)DeeployNetwork_input_10;
  DeeployNetwork_inputs[11] = (void *)DeeployNetwork_input_11;
  DeeployNetwork_inputs[12] = (void *)DeeployNetwork_input_12;
  DeeployNetwork_inputs[13] = (void *)DeeployNetwork_input_13;
  DeeployNetwork_inputs[14] = (void *)DeeployNetwork_input_14;
  DeeployNetwork_inputs[15] = (void *)DeeployNetwork_input_15;
  DeeployNetwork_inputs[16] = (void *)DeeployNetwork_input_16;
  DeeployNetwork_inputs[17] = (void *)DeeployNetwork_input_17;
  DeeployNetwork_inputs[18] = (void *)DeeployNetwork_input_18;
  DeeployNetwork_inputs[19] = (void *)DeeployNetwork_input_19;
  DeeployNetwork_inputs[20] = (void *)DeeployNetwork_input_20;
  DeeployNetwork_inputs[21] = (void *)DeeployNetwork_input_21;
  DeeployNetwork_inputs[22] = (void *)DeeployNetwork_input_22;
  DeeployNetwork_inputs[23] = (void *)DeeployNetwork_input_23;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
  DeeployNetwork_outputs[1] = (void *)DeeployNetwork_output_1;
}
