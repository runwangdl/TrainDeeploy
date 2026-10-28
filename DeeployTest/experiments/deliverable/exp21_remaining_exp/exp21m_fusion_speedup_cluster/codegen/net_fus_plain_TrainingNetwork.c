#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
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

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {421, 251, 363, 284, 342, 270, 321, 445};

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor[8] = {85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {16715, 28051, 19407, 24750, 20593, 26069, 21962, 15816};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_mean__dual6_tensor[16] = {
    -0.0019563890527933836f,  -4.265925963409245e-05f,  0.0010741822188720107f,  -0.00022483340580947697f, -3.8470159779535607e-05f, -0.00020210926595609635f,
    0.00030828602029941976f,  0.00018801260739564896f,  -0.0019563890527933836f, -4.265925963409245e-05f,  0.0010741822188720107f,   -0.00022483340580947697f,
    -3.8470159779535607e-05f, -0.00020210926595609635f, 0.00030828602029941976f, 0.00018801260739564896f};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_var__dual7_tensor[16] = {
    10794.6962890625f, 1960.42578125f, 1352.2359619140625f, 1431.591064453125f, 4565.24951171875f, 1289.480712890625f, 1300.5797119140625f, 14838.5048828125f,
    10794.6962890625f, 1960.42578125f, 1352.2359619140625f, 1431.591064453125f, 4565.24951171875f, 1289.480712890625f, 1300.5797119140625f, 14838.5048828125f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {69, 67, 67, 59, 60, 62, 64, 66,
                                                                                                                 74, 58, 69, 65, 60, 69, 60, 58};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor[16] = {234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192,
                                                                              219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {3887854, 4028793, 3998692, 4560022, 4527256, 4318277, 4195455, 4117846,
                                                                              3644722, 4679690, 3915633, 4146882, 4516702, 3937397, 4507614, 4631827};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_mean__dual13_tensor[32] = {
    -0.2444174885749817f,  0.21689638495445251f, 0.14622187614440918f, -1.0542712211608887f, -1.0498594045639038f, -0.22530169785022736f, -1.1160809993743896f,
    1.0315797328948975f,   1.0182175636291504f,  0.12470299005508423f, -1.3734524250030518f, -0.7460818886756897f, -0.670293390750885f,   -1.0082124471664429f,
    0.6750643253326416f,   0.8555498123168945f,  -0.2444174885749817f, 0.21689638495445251f, 0.14622187614440918f, -1.0542712211608887f,  -1.0498594045639038f,
    -0.22530169785022736f, -1.1160809993743896f, 1.0315797328948975f,  1.0182175636291504f,  0.12470299005508423f, -1.3734524250030518f,  -0.7460818886756897f,
    -0.670293390750885f,   -1.0082124471664429f, 0.6750643253326416f,  0.8555498123168945f};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_var__dual14_tensor[32] = {
    1.4679807424545288f, 1.4829299449920654f, 4.743453502655029f,  5.358211040496826f,  4.712860107421875f, 4.743685245513916f,  4.386803150177002f,
    2.8539552688598633f, 3.485712766647339f,  1.066118597984314f,  10.345640182495117f, 5.888366222381592f, 1.6831457614898682f, 3.925990581512451f,
    2.867671489715576f,  2.335334300994873f,  1.4679807424545288f, 1.4829299449920654f, 4.743453502655029f, 5.358211040496826f,  4.712860107421875f,
    4.743685245513916f,  4.386803150177002f,  2.8539552688598633f, 3.485712766647339f,  1.066118597984314f, 10.345640182495117f, 5.888366222381592f,
    1.6831457614898682f, 3.925990581512451f,  2.867671489715576f,  2.335334300994873f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {136, 115, 104, 112, 123, 116, 124, 117,
                                                                                                                 102, 165, 114, 129, 101, 128, 167, 97};

static PI_L2 int32_t DeeployNetwork_blocks2convweight_int8_pmul_tensor[16] = {187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670,
                                                                              249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor[16] = {5779922, 6811814, 7569997, 7017406, 6391308, 6797053, 6348470, 6703570,
                                                                              7695533, 4777151, 6888298, 6076617, 7800891, 6162310, 4713482, 8093983};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_mean__dual20_tensor[32] = {
    -0.4543471932411194f,  0.033739298582077026f,  -0.16740328073501587f, -0.633358359336853f,     0.06192547827959061f,  -0.29306620359420776f,
    -0.18620505928993225f, -0.16867128014564514f,  0.24714651703834534f,  -0.09643757343292236f,   -0.2440318465232849f,  -0.32666322588920593f,
    0.3093155026435852f,   -0.48663511872291565f,  -0.43303781747817993f, -0.0018389797769486904f, -0.4543471932411194f,  0.033739298582077026f,
    -0.16740328073501587f, -0.633358359336853f,    0.06192547827959061f,  -0.29306620359420776f,   -0.18620505928993225f, -0.16867128014564514f,
    0.24714651703834534f,  -0.09643757343292236f,  -0.2440318465232849f,  -0.32666322588920593f,   0.3093155026435852f,   -0.48663511872291565f,
    -0.43303781747817993f, -0.0018389797769486904f};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_var__dual21_tensor[32] = {
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

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_mean__dual27_tensor[64] = {
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

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_var__dual28_tensor[64] = {
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

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_mean__dual34_tensor[64] = {
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

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_var__dual35_tensor[64] = {
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

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_cumByteOffset[1] = {
        0};

static PI_L2 uint8_t
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_cumByteOffset[1] = {
        0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y[2] = {478, 225};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y[2] = {477, 224};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left[2] = {2, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right[2] = {0, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_numTiles[2] = {0, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_cumByteOffset[2] = {0, 475};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_cmd[2] = {1972772, 1969230};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_size_1d[2] = {478, 225};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_cumByteOffset[2] = {0, 0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_cumByteOffset[2] = {0, 7632};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_cmd[2] = {1941856, 1885184};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_size_1d[2] = {7632, 3584};

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

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_cumByteOffset[1] = {0};

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

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[24];
void *DeeployNetwork_outputs[2];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref;
} _pert_fcbias_fp32_mode1__dual44_tiling_closure_args_t;

static void _pert_fcbias_fp32_mode1__dual44_tiling_closure(void *_pert_fcbias_fp32_mode1__dual44_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual44_tiling_closure_args_t *args =
      (_pert_fcbias_fp32_mode1__dual44_tiling_closure_args_t *)_pert_fcbias_fp32_mode1__dual44_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcbias_fp32_mode1__dual44, Op: PerturbRademacherFixed, mode 1) -- QW exp8
  uint8_t pert_fcbias_fp32_mode1__dual44_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcbias_fp32_mode1__dual44_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcbias_fp32_mode1__dual44_chunk = (9 >> pert_fcbias_fp32_mode1__dual44_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcbias_fp32_mode1__dual44_chunk_start =
      (uint32_t)MIN(pert_fcbias_fp32_mode1__dual44_chunk * pert_fcbias_fp32_mode1__dual44_core_id, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode1__dual44_chunk_stop =
      (uint32_t)MIN(pert_fcbias_fp32_mode1__dual44_chunk_start + pert_fcbias_fp32_mode1__dual44_chunk, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode1__dual44_local_size = pert_fcbias_fp32_mode1__dual44_chunk_stop - pert_fcbias_fp32_mode1__dual44_chunk_start;
  uint32_t pert_fcbias_fp32_mode1__dual44_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_fcbias_fp32_mode1__dual44_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcbias_fp32_mode1__dual44_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref[pert_fcbias_fp32_mode1__dual44_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref)[pert_fcbias_fp32_mode1__dual44_chunk_start],
      pert_fcbias_fp32_mode1__dual44_seed, 1u, pert_fcbias_fp32_mode1__dual44_local_size, pert_fcbias_fp32_mode1__dual44_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref;
} _pert_fcbias_fp32_mode1__dual44_cluster_fork_args_t;

static void _pert_fcbias_fp32_mode1__dual44_cluster_fork(void *_pert_fcbias_fp32_mode1__dual44_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual44_cluster_fork_args_t *args =
      (_pert_fcbias_fp32_mode1__dual44_cluster_fork_args_t *)_pert_fcbias_fp32_mode1__dual44_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode1__dual44_tiling_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual44_tiling_closure_args =
      (_pert_fcbias_fp32_mode1__dual44_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref};

  // _pert_fcbias_fp32_mode1__dual44_tiling_closure CLOSURE CALL
  _pert_fcbias_fp32_mode1__dual44_tiling_closure(&DeeployNetwork__pert_fcbias_fp32_mode1__dual44_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr;
} _pert_fcbias_fp32_mode1__dual44_closure_args_t;

static void _pert_fcbias_fp32_mode1__dual44_closure(void *_pert_fcbias_fp32_mode1__dual44_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual44_closure_args_t *args = (_pert_fcbias_fp32_mode1__dual44_closure_args_t *)_pert_fcbias_fp32_mode1__dual44_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_input_23_ref =
      (void *)((char *)DeeployNetwork_input_23 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_fcbias_fp32_pert_p__dual43_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_input_23_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcbias_fp32_mode1__dual44_cluster_fork_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual44_cluster_fork_args =
        (_pert_fcbias_fp32_mode1__dual44_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcbias_fp32_mode1__dual44_cluster_fork, &DeeployNetwork__pert_fcbias_fp32_mode1__dual44_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_fcbias_fp32_pert_p__dual43_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr;
} _pert_fcbias_fp32_mode1__dual44_closure_L3_args_t;

static void _pert_fcbias_fp32_mode1__dual44_closure_L3(void *_pert_fcbias_fp32_mode1__dual44_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode1__dual44_closure_L3_args_t *args =
      (_pert_fcbias_fp32_mode1__dual44_closure_L3_args_t *)_pert_fcbias_fp32_mode1__dual44_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode1__dual44_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual44_closure_args = (_pert_fcbias_fp32_mode1__dual44_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr};

  // _pert_fcbias_fp32_mode1__dual44_closure CLOSURE CALL
  _pert_fcbias_fp32_mode1__dual44_closure(&DeeployNetwork__pert_fcbias_fp32_mode1__dual44_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref;
} _pert_fcbias_fp32_mode2__dual50_tiling_closure_args_t;

static void _pert_fcbias_fp32_mode2__dual50_tiling_closure(void *_pert_fcbias_fp32_mode2__dual50_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual50_tiling_closure_args_t *args =
      (_pert_fcbias_fp32_mode2__dual50_tiling_closure_args_t *)_pert_fcbias_fp32_mode2__dual50_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcbias_fp32_mode2__dual50, Op: PerturbRademacherFixed, mode 2) -- QW exp8
  uint8_t pert_fcbias_fp32_mode2__dual50_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcbias_fp32_mode2__dual50_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcbias_fp32_mode2__dual50_chunk = (9 >> pert_fcbias_fp32_mode2__dual50_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcbias_fp32_mode2__dual50_chunk_start =
      (uint32_t)MIN(pert_fcbias_fp32_mode2__dual50_chunk * pert_fcbias_fp32_mode2__dual50_core_id, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode2__dual50_chunk_stop =
      (uint32_t)MIN(pert_fcbias_fp32_mode2__dual50_chunk_start + pert_fcbias_fp32_mode2__dual50_chunk, (uint32_t)9);
  uint32_t pert_fcbias_fp32_mode2__dual50_local_size = pert_fcbias_fp32_mode2__dual50_chunk_stop - pert_fcbias_fp32_mode2__dual50_chunk_start;
  uint32_t pert_fcbias_fp32_mode2__dual50_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_fcbias_fp32_mode2__dual50_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcbias_fp32_mode2__dual50_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref[pert_fcbias_fp32_mode2__dual50_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref)[pert_fcbias_fp32_mode2__dual50_chunk_start],
      pert_fcbias_fp32_mode2__dual50_seed, 0u, pert_fcbias_fp32_mode2__dual50_local_size, pert_fcbias_fp32_mode2__dual50_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref;
} _pert_fcbias_fp32_mode2__dual50_cluster_fork_args_t;

static void _pert_fcbias_fp32_mode2__dual50_cluster_fork(void *_pert_fcbias_fp32_mode2__dual50_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual50_cluster_fork_args_t *args =
      (_pert_fcbias_fp32_mode2__dual50_cluster_fork_args_t *)_pert_fcbias_fp32_mode2__dual50_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode2__dual50_tiling_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual50_tiling_closure_args =
      (_pert_fcbias_fp32_mode2__dual50_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref};

  // _pert_fcbias_fp32_mode2__dual50_tiling_closure CLOSURE CALL
  _pert_fcbias_fp32_mode2__dual50_tiling_closure(&DeeployNetwork__pert_fcbias_fp32_mode2__dual50_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr;
} _pert_fcbias_fp32_mode2__dual50_closure_args_t;

static void _pert_fcbias_fp32_mode2__dual50_closure(void *_pert_fcbias_fp32_mode2__dual50_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual50_closure_args_t *args = (_pert_fcbias_fp32_mode2__dual50_closure_args_t *)_pert_fcbias_fp32_mode2__dual50_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_input_23_ref =
      (void *)((char *)DeeployNetwork_input_23 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_fcbias_fp32_pert_m__dual49_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor + DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_input_23_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcbias_fp32_mode2__dual50_cluster_fork_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual50_cluster_fork_args =
        (_pert_fcbias_fp32_mode2__dual50_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_in_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcbias_fp32_mode2__dual50_cluster_fork, &DeeployNetwork__pert_fcbias_fp32_mode2__dual50_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_fcbias_fp32_pert_m__dual49_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr;
} _pert_fcbias_fp32_mode2__dual50_closure_L3_args_t;

static void _pert_fcbias_fp32_mode2__dual50_closure_L3(void *_pert_fcbias_fp32_mode2__dual50_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_mode2__dual50_closure_L3_args_t *args =
      (_pert_fcbias_fp32_mode2__dual50_closure_L3_args_t *)_pert_fcbias_fp32_mode2__dual50_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_mode2__dual50_closure_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual50_closure_args = (_pert_fcbias_fp32_mode2__dual50_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr};

  // _pert_fcbias_fp32_mode2__dual50_closure CLOSURE CALL
  _pert_fcbias_fp32_mode2__dual50_closure(&DeeployNetwork__pert_fcbias_fp32_mode2__dual50_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref;
} _pert_blocks0bnweight_mode0__dual3_tiling_closure_args_t;

static void _pert_blocks0bnweight_mode0__dual3_tiling_closure(void *_pert_blocks0bnweight_mode0__dual3_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual3_tiling_closure_args_t *args =
      (_pert_blocks0bnweight_mode0__dual3_tiling_closure_args_t *)_pert_blocks0bnweight_mode0__dual3_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks0bnweight_mode0__dual3, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks0bnweight_mode0__dual3_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnweight_mode0__dual3_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnweight_mode0__dual3_chunk = (8 >> pert_blocks0bnweight_mode0__dual3_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnweight_mode0__dual3_chunk_start =
      (uint32_t)MIN(pert_blocks0bnweight_mode0__dual3_chunk * pert_blocks0bnweight_mode0__dual3_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnweight_mode0__dual3_chunk_stop =
      (uint32_t)MIN(pert_blocks0bnweight_mode0__dual3_chunk_start + pert_blocks0bnweight_mode0__dual3_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnweight_mode0__dual3_local_size = pert_blocks0bnweight_mode0__dual3_chunk_stop - pert_blocks0bnweight_mode0__dual3_chunk_start;
  uint32_t pert_blocks0bnweight_mode0__dual3_seed = ((42 + perturb_seed_base) + NUM_CORES * 12 + pert_blocks0bnweight_mode0__dual3_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnweight_mode0__dual3_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref[pert_blocks0bnweight_mode0__dual3_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref)[pert_blocks0bnweight_mode0__dual3_chunk_start],
      pert_blocks0bnweight_mode0__dual3_seed, 1u, pert_blocks0bnweight_mode0__dual3_local_size, pert_blocks0bnweight_mode0__dual3_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref[pert_blocks0bnweight_mode0__dual3_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref[8 + pert_blocks0bnweight_mode0__dual3_chunk_start],
      pert_blocks0bnweight_mode0__dual3_seed, 0u, pert_blocks0bnweight_mode0__dual3_local_size, pert_blocks0bnweight_mode0__dual3_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref;
} _pert_blocks0bnweight_mode0__dual3_cluster_fork_args_t;

static void _pert_blocks0bnweight_mode0__dual3_cluster_fork(void *_pert_blocks0bnweight_mode0__dual3_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual3_cluster_fork_args_t *args =
      (_pert_blocks0bnweight_mode0__dual3_cluster_fork_args_t *)_pert_blocks0bnweight_mode0__dual3_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_mode0__dual3_tiling_closure_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual3_tiling_closure_args =
      (_pert_blocks0bnweight_mode0__dual3_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref,
                                                                 .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref};

  // _pert_blocks0bnweight_mode0__dual3_tiling_closure CLOSURE CALL
  _pert_blocks0bnweight_mode0__dual3_tiling_closure(&DeeployNetwork__pert_blocks0bnweight_mode0__dual3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr;
} _pert_blocks0bnweight_mode0__dual3_closure_args_t;

static void _pert_blocks0bnweight_mode0__dual3_closure(void *_pert_blocks0bnweight_mode0__dual3_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual3_closure_args_t *args =
      (_pert_blocks0bnweight_mode0__dual3_closure_args_t *)_pert_blocks0bnweight_mode0__dual3_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_blocks0bnweight_pert_pm__dual2_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_input_2_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks0bnweight_mode0__dual3_cluster_fork_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual3_cluster_fork_args =
        (_pert_blocks0bnweight_mode0__dual3_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_in_ref,
                                                                 .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnweight_mode0__dual3_cluster_fork, &DeeployNetwork__pert_blocks0bnweight_mode0__dual3_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_blocks0bnweight_pert_pm__dual2_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr;
} _pert_blocks0bnweight_mode0__dual3_closure_L3_args_t;

static void _pert_blocks0bnweight_mode0__dual3_closure_L3(void *_pert_blocks0bnweight_mode0__dual3_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_mode0__dual3_closure_L3_args_t *args =
      (_pert_blocks0bnweight_mode0__dual3_closure_L3_args_t *)_pert_blocks0bnweight_mode0__dual3_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_mode0__dual3_closure_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual3_closure_args =
      (_pert_blocks0bnweight_mode0__dual3_closure_args_t){.DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor =
                                                              DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor,
                                                          .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr =
                                                              DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr};

  // _pert_blocks0bnweight_mode0__dual3_closure CLOSURE CALL
  _pert_blocks0bnweight_mode0__dual3_closure(&DeeployNetwork__pert_blocks0bnweight_mode0__dual3_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref;
} _pert_blocks0bnbias_mode0__dual5_tiling_closure_args_t;

static void _pert_blocks0bnbias_mode0__dual5_tiling_closure(void *_pert_blocks0bnbias_mode0__dual5_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual5_tiling_closure_args_t *args =
      (_pert_blocks0bnbias_mode0__dual5_tiling_closure_args_t *)_pert_blocks0bnbias_mode0__dual5_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks0bnbias_mode0__dual5, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks0bnbias_mode0__dual5_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnbias_mode0__dual5_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnbias_mode0__dual5_chunk = (8 >> pert_blocks0bnbias_mode0__dual5_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnbias_mode0__dual5_chunk_start =
      (uint32_t)MIN(pert_blocks0bnbias_mode0__dual5_chunk * pert_blocks0bnbias_mode0__dual5_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnbias_mode0__dual5_chunk_stop =
      (uint32_t)MIN(pert_blocks0bnbias_mode0__dual5_chunk_start + pert_blocks0bnbias_mode0__dual5_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnbias_mode0__dual5_local_size = pert_blocks0bnbias_mode0__dual5_chunk_stop - pert_blocks0bnbias_mode0__dual5_chunk_start;
  uint32_t pert_blocks0bnbias_mode0__dual5_seed = ((42 + perturb_seed_base) + NUM_CORES * 13 + pert_blocks0bnbias_mode0__dual5_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnbias_mode0__dual5_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref[pert_blocks0bnbias_mode0__dual5_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref)[pert_blocks0bnbias_mode0__dual5_chunk_start],
      pert_blocks0bnbias_mode0__dual5_seed, 1u, pert_blocks0bnbias_mode0__dual5_local_size, pert_blocks0bnbias_mode0__dual5_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref[pert_blocks0bnbias_mode0__dual5_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref[8 + pert_blocks0bnbias_mode0__dual5_chunk_start],
      pert_blocks0bnbias_mode0__dual5_seed, 0u, pert_blocks0bnbias_mode0__dual5_local_size, pert_blocks0bnbias_mode0__dual5_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref;
} _pert_blocks0bnbias_mode0__dual5_cluster_fork_args_t;

static void _pert_blocks0bnbias_mode0__dual5_cluster_fork(void *_pert_blocks0bnbias_mode0__dual5_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual5_cluster_fork_args_t *args =
      (_pert_blocks0bnbias_mode0__dual5_cluster_fork_args_t *)_pert_blocks0bnbias_mode0__dual5_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_mode0__dual5_tiling_closure_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual5_tiling_closure_args =
      (_pert_blocks0bnbias_mode0__dual5_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref};

  // _pert_blocks0bnbias_mode0__dual5_tiling_closure CLOSURE CALL
  _pert_blocks0bnbias_mode0__dual5_tiling_closure(&DeeployNetwork__pert_blocks0bnbias_mode0__dual5_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr;
} _pert_blocks0bnbias_mode0__dual5_closure_args_t;

static void _pert_blocks0bnbias_mode0__dual5_closure(void *_pert_blocks0bnbias_mode0__dual5_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual5_closure_args_t *args = (_pert_blocks0bnbias_mode0__dual5_closure_args_t *)_pert_blocks0bnbias_mode0__dual5_closure_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_blocks0bnbias_pert_pm__dual4_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks0bnbias_mode0__dual5_cluster_fork_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual5_cluster_fork_args =
        (_pert_blocks0bnbias_mode0__dual5_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_in_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnbias_mode0__dual5_cluster_fork, &DeeployNetwork__pert_blocks0bnbias_mode0__dual5_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_blocks0bnbias_pert_pm__dual4_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr;
} _pert_blocks0bnbias_mode0__dual5_closure_L3_args_t;

static void _pert_blocks0bnbias_mode0__dual5_closure_L3(void *_pert_blocks0bnbias_mode0__dual5_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_mode0__dual5_closure_L3_args_t *args =
      (_pert_blocks0bnbias_mode0__dual5_closure_L3_args_t *)_pert_blocks0bnbias_mode0__dual5_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_mode0__dual5_closure_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual5_closure_args =
      (_pert_blocks0bnbias_mode0__dual5_closure_args_t){.DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor =
                                                            DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor,
                                                        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr =
                                                            DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr};

  // _pert_blocks0bnbias_mode0__dual5_closure CLOSURE CALL
  _pert_blocks0bnbias_mode0__dual5_closure(&DeeployNetwork__pert_blocks0bnbias_mode0__dual5_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref;
} _pert_blocks1bnweight_mode0__dual10_tiling_closure_args_t;

static void _pert_blocks1bnweight_mode0__dual10_tiling_closure(void *_pert_blocks1bnweight_mode0__dual10_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual10_tiling_closure_args_t *args =
      (_pert_blocks1bnweight_mode0__dual10_tiling_closure_args_t *)_pert_blocks1bnweight_mode0__dual10_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks1bnweight_mode0__dual10, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks1bnweight_mode0__dual10_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnweight_mode0__dual10_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnweight_mode0__dual10_chunk = (16 >> pert_blocks1bnweight_mode0__dual10_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnweight_mode0__dual10_chunk_start =
      (uint32_t)MIN(pert_blocks1bnweight_mode0__dual10_chunk * pert_blocks1bnweight_mode0__dual10_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnweight_mode0__dual10_chunk_stop =
      (uint32_t)MIN(pert_blocks1bnweight_mode0__dual10_chunk_start + pert_blocks1bnweight_mode0__dual10_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnweight_mode0__dual10_local_size = pert_blocks1bnweight_mode0__dual10_chunk_stop - pert_blocks1bnweight_mode0__dual10_chunk_start;
  uint32_t pert_blocks1bnweight_mode0__dual10_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 14 + pert_blocks1bnweight_mode0__dual10_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnweight_mode0__dual10_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref[pert_blocks1bnweight_mode0__dual10_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref)[pert_blocks1bnweight_mode0__dual10_chunk_start],
      pert_blocks1bnweight_mode0__dual10_seed, 1u, pert_blocks1bnweight_mode0__dual10_local_size, pert_blocks1bnweight_mode0__dual10_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref[pert_blocks1bnweight_mode0__dual10_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref[16 + pert_blocks1bnweight_mode0__dual10_chunk_start],
      pert_blocks1bnweight_mode0__dual10_seed, 0u, pert_blocks1bnweight_mode0__dual10_local_size, pert_blocks1bnweight_mode0__dual10_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref;
} _pert_blocks1bnweight_mode0__dual10_cluster_fork_args_t;

static void _pert_blocks1bnweight_mode0__dual10_cluster_fork(void *_pert_blocks1bnweight_mode0__dual10_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual10_cluster_fork_args_t *args =
      (_pert_blocks1bnweight_mode0__dual10_cluster_fork_args_t *)_pert_blocks1bnweight_mode0__dual10_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_mode0__dual10_tiling_closure_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual10_tiling_closure_args =
      (_pert_blocks1bnweight_mode0__dual10_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref};

  // _pert_blocks1bnweight_mode0__dual10_tiling_closure CLOSURE CALL
  _pert_blocks1bnweight_mode0__dual10_tiling_closure(&DeeployNetwork__pert_blocks1bnweight_mode0__dual10_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr;
} _pert_blocks1bnweight_mode0__dual10_closure_args_t;

static void _pert_blocks1bnweight_mode0__dual10_closure(void *_pert_blocks1bnweight_mode0__dual10_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual10_closure_args_t *args =
      (_pert_blocks1bnweight_mode0__dual10_closure_args_t *)_pert_blocks1bnweight_mode0__dual10_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_input_4_ref =
      (void *)((char *)DeeployNetwork_input_4 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_blocks1bnweight_pert_pm__dual9_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_input_4_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks1bnweight_mode0__dual10_cluster_fork_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual10_cluster_fork_args =
        (_pert_blocks1bnweight_mode0__dual10_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnweight_mode0__dual10_cluster_fork, &DeeployNetwork__pert_blocks1bnweight_mode0__dual10_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_blocks1bnweight_pert_pm__dual9_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr;
} _pert_blocks1bnweight_mode0__dual10_closure_L3_args_t;

static void _pert_blocks1bnweight_mode0__dual10_closure_L3(void *_pert_blocks1bnweight_mode0__dual10_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_mode0__dual10_closure_L3_args_t *args =
      (_pert_blocks1bnweight_mode0__dual10_closure_L3_args_t *)_pert_blocks1bnweight_mode0__dual10_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_mode0__dual10_closure_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual10_closure_args =
      (_pert_blocks1bnweight_mode0__dual10_closure_args_t){.DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor =
                                                               DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr};

  // _pert_blocks1bnweight_mode0__dual10_closure CLOSURE CALL
  _pert_blocks1bnweight_mode0__dual10_closure(&DeeployNetwork__pert_blocks1bnweight_mode0__dual10_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref;
} _pert_blocks1bnbias_mode0__dual12_tiling_closure_args_t;

static void _pert_blocks1bnbias_mode0__dual12_tiling_closure(void *_pert_blocks1bnbias_mode0__dual12_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual12_tiling_closure_args_t *args =
      (_pert_blocks1bnbias_mode0__dual12_tiling_closure_args_t *)_pert_blocks1bnbias_mode0__dual12_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks1bnbias_mode0__dual12, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks1bnbias_mode0__dual12_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnbias_mode0__dual12_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnbias_mode0__dual12_chunk = (16 >> pert_blocks1bnbias_mode0__dual12_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnbias_mode0__dual12_chunk_start =
      (uint32_t)MIN(pert_blocks1bnbias_mode0__dual12_chunk * pert_blocks1bnbias_mode0__dual12_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnbias_mode0__dual12_chunk_stop =
      (uint32_t)MIN(pert_blocks1bnbias_mode0__dual12_chunk_start + pert_blocks1bnbias_mode0__dual12_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnbias_mode0__dual12_local_size = pert_blocks1bnbias_mode0__dual12_chunk_stop - pert_blocks1bnbias_mode0__dual12_chunk_start;
  uint32_t pert_blocks1bnbias_mode0__dual12_seed = ((42 + perturb_seed_base) + NUM_CORES * 15 + pert_blocks1bnbias_mode0__dual12_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnbias_mode0__dual12_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref[pert_blocks1bnbias_mode0__dual12_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref)[pert_blocks1bnbias_mode0__dual12_chunk_start],
      pert_blocks1bnbias_mode0__dual12_seed, 1u, pert_blocks1bnbias_mode0__dual12_local_size, pert_blocks1bnbias_mode0__dual12_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref[pert_blocks1bnbias_mode0__dual12_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref[16 + pert_blocks1bnbias_mode0__dual12_chunk_start],
      pert_blocks1bnbias_mode0__dual12_seed, 0u, pert_blocks1bnbias_mode0__dual12_local_size, pert_blocks1bnbias_mode0__dual12_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref;
} _pert_blocks1bnbias_mode0__dual12_cluster_fork_args_t;

static void _pert_blocks1bnbias_mode0__dual12_cluster_fork(void *_pert_blocks1bnbias_mode0__dual12_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual12_cluster_fork_args_t *args =
      (_pert_blocks1bnbias_mode0__dual12_cluster_fork_args_t *)_pert_blocks1bnbias_mode0__dual12_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_mode0__dual12_tiling_closure_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual12_tiling_closure_args =
      (_pert_blocks1bnbias_mode0__dual12_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref};

  // _pert_blocks1bnbias_mode0__dual12_tiling_closure CLOSURE CALL
  _pert_blocks1bnbias_mode0__dual12_tiling_closure(&DeeployNetwork__pert_blocks1bnbias_mode0__dual12_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr;
} _pert_blocks1bnbias_mode0__dual12_closure_args_t;

static void _pert_blocks1bnbias_mode0__dual12_closure(void *_pert_blocks1bnbias_mode0__dual12_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual12_closure_args_t *args = (_pert_blocks1bnbias_mode0__dual12_closure_args_t *)_pert_blocks1bnbias_mode0__dual12_closure_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_input_5_ref =
      (void *)((char *)DeeployNetwork_input_5 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_blocks1bnbias_pert_pm__dual11_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_input_5_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks1bnbias_mode0__dual12_cluster_fork_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual12_cluster_fork_args =
        (_pert_blocks1bnbias_mode0__dual12_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnbias_mode0__dual12_cluster_fork, &DeeployNetwork__pert_blocks1bnbias_mode0__dual12_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_blocks1bnbias_pert_pm__dual11_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr;
} _pert_blocks1bnbias_mode0__dual12_closure_L3_args_t;

static void _pert_blocks1bnbias_mode0__dual12_closure_L3(void *_pert_blocks1bnbias_mode0__dual12_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_mode0__dual12_closure_L3_args_t *args =
      (_pert_blocks1bnbias_mode0__dual12_closure_L3_args_t *)_pert_blocks1bnbias_mode0__dual12_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_mode0__dual12_closure_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual12_closure_args =
      (_pert_blocks1bnbias_mode0__dual12_closure_args_t){.DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor =
                                                             DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr};

  // _pert_blocks1bnbias_mode0__dual12_closure CLOSURE CALL
  _pert_blocks1bnbias_mode0__dual12_closure(&DeeployNetwork__pert_blocks1bnbias_mode0__dual12_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref;
} _pert_blocks2bnweight_mode0__dual17_tiling_closure_args_t;

static void _pert_blocks2bnweight_mode0__dual17_tiling_closure(void *_pert_blocks2bnweight_mode0__dual17_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual17_tiling_closure_args_t *args =
      (_pert_blocks2bnweight_mode0__dual17_tiling_closure_args_t *)_pert_blocks2bnweight_mode0__dual17_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks2bnweight_mode0__dual17, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks2bnweight_mode0__dual17_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnweight_mode0__dual17_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnweight_mode0__dual17_chunk = (16 >> pert_blocks2bnweight_mode0__dual17_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnweight_mode0__dual17_chunk_start =
      (uint32_t)MIN(pert_blocks2bnweight_mode0__dual17_chunk * pert_blocks2bnweight_mode0__dual17_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnweight_mode0__dual17_chunk_stop =
      (uint32_t)MIN(pert_blocks2bnweight_mode0__dual17_chunk_start + pert_blocks2bnweight_mode0__dual17_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnweight_mode0__dual17_local_size = pert_blocks2bnweight_mode0__dual17_chunk_stop - pert_blocks2bnweight_mode0__dual17_chunk_start;
  uint32_t pert_blocks2bnweight_mode0__dual17_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 16 + pert_blocks2bnweight_mode0__dual17_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnweight_mode0__dual17_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref[pert_blocks2bnweight_mode0__dual17_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref)[pert_blocks2bnweight_mode0__dual17_chunk_start],
      pert_blocks2bnweight_mode0__dual17_seed, 1u, pert_blocks2bnweight_mode0__dual17_local_size, pert_blocks2bnweight_mode0__dual17_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref[pert_blocks2bnweight_mode0__dual17_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref[16 + pert_blocks2bnweight_mode0__dual17_chunk_start],
      pert_blocks2bnweight_mode0__dual17_seed, 0u, pert_blocks2bnweight_mode0__dual17_local_size, pert_blocks2bnweight_mode0__dual17_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref;
} _pert_blocks2bnweight_mode0__dual17_cluster_fork_args_t;

static void _pert_blocks2bnweight_mode0__dual17_cluster_fork(void *_pert_blocks2bnweight_mode0__dual17_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual17_cluster_fork_args_t *args =
      (_pert_blocks2bnweight_mode0__dual17_cluster_fork_args_t *)_pert_blocks2bnweight_mode0__dual17_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_mode0__dual17_tiling_closure_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual17_tiling_closure_args =
      (_pert_blocks2bnweight_mode0__dual17_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref};

  // _pert_blocks2bnweight_mode0__dual17_tiling_closure CLOSURE CALL
  _pert_blocks2bnweight_mode0__dual17_tiling_closure(&DeeployNetwork__pert_blocks2bnweight_mode0__dual17_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr;
} _pert_blocks2bnweight_mode0__dual17_closure_args_t;

static void _pert_blocks2bnweight_mode0__dual17_closure(void *_pert_blocks2bnweight_mode0__dual17_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual17_closure_args_t *args =
      (_pert_blocks2bnweight_mode0__dual17_closure_args_t *)_pert_blocks2bnweight_mode0__dual17_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_input_6_ref =
      (void *)((char *)DeeployNetwork_input_6 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_blocks2bnweight_pert_pm__dual16_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_input_6_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks2bnweight_mode0__dual17_cluster_fork_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual17_cluster_fork_args =
        (_pert_blocks2bnweight_mode0__dual17_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnweight_mode0__dual17_cluster_fork, &DeeployNetwork__pert_blocks2bnweight_mode0__dual17_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_blocks2bnweight_pert_pm__dual16_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr;
} _pert_blocks2bnweight_mode0__dual17_closure_L3_args_t;

static void _pert_blocks2bnweight_mode0__dual17_closure_L3(void *_pert_blocks2bnweight_mode0__dual17_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_mode0__dual17_closure_L3_args_t *args =
      (_pert_blocks2bnweight_mode0__dual17_closure_L3_args_t *)_pert_blocks2bnweight_mode0__dual17_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_mode0__dual17_closure_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual17_closure_args =
      (_pert_blocks2bnweight_mode0__dual17_closure_args_t){.DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor =
                                                               DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr};

  // _pert_blocks2bnweight_mode0__dual17_closure CLOSURE CALL
  _pert_blocks2bnweight_mode0__dual17_closure(&DeeployNetwork__pert_blocks2bnweight_mode0__dual17_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref;
} _pert_blocks2bnbias_mode0__dual19_tiling_closure_args_t;

static void _pert_blocks2bnbias_mode0__dual19_tiling_closure(void *_pert_blocks2bnbias_mode0__dual19_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual19_tiling_closure_args_t *args =
      (_pert_blocks2bnbias_mode0__dual19_tiling_closure_args_t *)_pert_blocks2bnbias_mode0__dual19_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks2bnbias_mode0__dual19, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks2bnbias_mode0__dual19_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnbias_mode0__dual19_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnbias_mode0__dual19_chunk = (16 >> pert_blocks2bnbias_mode0__dual19_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnbias_mode0__dual19_chunk_start =
      (uint32_t)MIN(pert_blocks2bnbias_mode0__dual19_chunk * pert_blocks2bnbias_mode0__dual19_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnbias_mode0__dual19_chunk_stop =
      (uint32_t)MIN(pert_blocks2bnbias_mode0__dual19_chunk_start + pert_blocks2bnbias_mode0__dual19_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnbias_mode0__dual19_local_size = pert_blocks2bnbias_mode0__dual19_chunk_stop - pert_blocks2bnbias_mode0__dual19_chunk_start;
  uint32_t pert_blocks2bnbias_mode0__dual19_seed = ((42 + perturb_seed_base) + NUM_CORES * 17 + pert_blocks2bnbias_mode0__dual19_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnbias_mode0__dual19_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref[pert_blocks2bnbias_mode0__dual19_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref)[pert_blocks2bnbias_mode0__dual19_chunk_start],
      pert_blocks2bnbias_mode0__dual19_seed, 1u, pert_blocks2bnbias_mode0__dual19_local_size, pert_blocks2bnbias_mode0__dual19_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref[pert_blocks2bnbias_mode0__dual19_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref[16 + pert_blocks2bnbias_mode0__dual19_chunk_start],
      pert_blocks2bnbias_mode0__dual19_seed, 0u, pert_blocks2bnbias_mode0__dual19_local_size, pert_blocks2bnbias_mode0__dual19_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref;
} _pert_blocks2bnbias_mode0__dual19_cluster_fork_args_t;

static void _pert_blocks2bnbias_mode0__dual19_cluster_fork(void *_pert_blocks2bnbias_mode0__dual19_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual19_cluster_fork_args_t *args =
      (_pert_blocks2bnbias_mode0__dual19_cluster_fork_args_t *)_pert_blocks2bnbias_mode0__dual19_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_mode0__dual19_tiling_closure_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual19_tiling_closure_args =
      (_pert_blocks2bnbias_mode0__dual19_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref};

  // _pert_blocks2bnbias_mode0__dual19_tiling_closure CLOSURE CALL
  _pert_blocks2bnbias_mode0__dual19_tiling_closure(&DeeployNetwork__pert_blocks2bnbias_mode0__dual19_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr;
} _pert_blocks2bnbias_mode0__dual19_closure_args_t;

static void _pert_blocks2bnbias_mode0__dual19_closure(void *_pert_blocks2bnbias_mode0__dual19_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual19_closure_args_t *args = (_pert_blocks2bnbias_mode0__dual19_closure_args_t *)_pert_blocks2bnbias_mode0__dual19_closure_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_input_7_ref =
      (void *)((char *)DeeployNetwork_input_7 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_blocks2bnbias_pert_pm__dual18_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_input_7_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks2bnbias_mode0__dual19_cluster_fork_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual19_cluster_fork_args =
        (_pert_blocks2bnbias_mode0__dual19_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnbias_mode0__dual19_cluster_fork, &DeeployNetwork__pert_blocks2bnbias_mode0__dual19_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_blocks2bnbias_pert_pm__dual18_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr;
} _pert_blocks2bnbias_mode0__dual19_closure_L3_args_t;

static void _pert_blocks2bnbias_mode0__dual19_closure_L3(void *_pert_blocks2bnbias_mode0__dual19_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_mode0__dual19_closure_L3_args_t *args =
      (_pert_blocks2bnbias_mode0__dual19_closure_L3_args_t *)_pert_blocks2bnbias_mode0__dual19_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_mode0__dual19_closure_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual19_closure_args =
      (_pert_blocks2bnbias_mode0__dual19_closure_args_t){.DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor =
                                                             DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr};

  // _pert_blocks2bnbias_mode0__dual19_closure CLOSURE CALL
  _pert_blocks2bnbias_mode0__dual19_closure(&DeeployNetwork__pert_blocks2bnbias_mode0__dual19_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args_t;

static void
_wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure(void *_wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [8, 1, 1, 4] -> [8, 1, 4, 1] (Name: wrappedInnerForwardImplConv__dual1_input_12_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 8;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_1;
  const uint32_t sOut_1 = sOut_2 * dimLen_3;
  const uint32_t sOut_0 = sOut_1 * dimLen_2;

  const uint32_t baseChunk_0 = dimLen_0 / NUM_CORES;
  const uint32_t leftover_0 = dimLen_0 - baseChunk_0 * NUM_CORES;
  const uint32_t offset_0 = baseChunk_0 * coreId + (coreId < leftover_0 ? coreId : leftover_0);
  const uint32_t chunk_0 = coreId < leftover_0 ? baseChunk_0 + 1 : baseChunk_0;
  for (uint32_t i_0 = offset_0; i_0 < offset_0 + chunk_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
      const uint32_t offS_1 = offS_0 + i_2 * sIn_2;
      const uint32_t offD_1 = offD_0 + i_2 * sOut_1;

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {
        const uint32_t offS_2 = offS_1 + i_3 * sIn_3;
        const uint32_t offD_2 = offD_1 + i_3 * sOut_2;

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
          const uint32_t offS_3 = offS_2 + i_1 * sIn_1;
          const uint32_t offD_3 = offD_2 + i_1 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args_t;

static void
_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork(void *_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args_t *)_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args =
          (_wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref};

  // _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args_t;

static void _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure(void *_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args_t *)_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_input_12_ref =
      (void *)((char *)DeeployNetwork_input_12 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_wrappedInnerForwardImplConv__dual1_input_12_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_input_12_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args =
            (_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1310752,
          .size = 32,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_wrappedInnerForwardImplConv__dual1_input_12_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args_t;

static void _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3(void *_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args_t *)_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args =
      (_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr};

  // _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure(&DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref;
} _pert_blocks3bnweight_mode0__dual24_tiling_closure_args_t;

static void _pert_blocks3bnweight_mode0__dual24_tiling_closure(void *_pert_blocks3bnweight_mode0__dual24_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual24_tiling_closure_args_t *args =
      (_pert_blocks3bnweight_mode0__dual24_tiling_closure_args_t *)_pert_blocks3bnweight_mode0__dual24_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks3bnweight_mode0__dual24, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks3bnweight_mode0__dual24_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnweight_mode0__dual24_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnweight_mode0__dual24_chunk = (32 >> pert_blocks3bnweight_mode0__dual24_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnweight_mode0__dual24_chunk_start =
      (uint32_t)MIN(pert_blocks3bnweight_mode0__dual24_chunk * pert_blocks3bnweight_mode0__dual24_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnweight_mode0__dual24_chunk_stop =
      (uint32_t)MIN(pert_blocks3bnweight_mode0__dual24_chunk_start + pert_blocks3bnweight_mode0__dual24_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnweight_mode0__dual24_local_size = pert_blocks3bnweight_mode0__dual24_chunk_stop - pert_blocks3bnweight_mode0__dual24_chunk_start;
  uint32_t pert_blocks3bnweight_mode0__dual24_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 18 + pert_blocks3bnweight_mode0__dual24_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnweight_mode0__dual24_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref[pert_blocks3bnweight_mode0__dual24_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref)[pert_blocks3bnweight_mode0__dual24_chunk_start],
      pert_blocks3bnweight_mode0__dual24_seed, 1u, pert_blocks3bnweight_mode0__dual24_local_size, pert_blocks3bnweight_mode0__dual24_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref[pert_blocks3bnweight_mode0__dual24_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref[32 + pert_blocks3bnweight_mode0__dual24_chunk_start],
      pert_blocks3bnweight_mode0__dual24_seed, 0u, pert_blocks3bnweight_mode0__dual24_local_size, pert_blocks3bnweight_mode0__dual24_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref;
} _pert_blocks3bnweight_mode0__dual24_cluster_fork_args_t;

static void _pert_blocks3bnweight_mode0__dual24_cluster_fork(void *_pert_blocks3bnweight_mode0__dual24_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual24_cluster_fork_args_t *args =
      (_pert_blocks3bnweight_mode0__dual24_cluster_fork_args_t *)_pert_blocks3bnweight_mode0__dual24_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_mode0__dual24_tiling_closure_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual24_tiling_closure_args =
      (_pert_blocks3bnweight_mode0__dual24_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref};

  // _pert_blocks3bnweight_mode0__dual24_tiling_closure CLOSURE CALL
  _pert_blocks3bnweight_mode0__dual24_tiling_closure(&DeeployNetwork__pert_blocks3bnweight_mode0__dual24_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr;
} _pert_blocks3bnweight_mode0__dual24_closure_args_t;

static void _pert_blocks3bnweight_mode0__dual24_closure(void *_pert_blocks3bnweight_mode0__dual24_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual24_closure_args_t *args =
      (_pert_blocks3bnweight_mode0__dual24_closure_args_t *)_pert_blocks3bnweight_mode0__dual24_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_input_8_ref =
      (void *)((char *)DeeployNetwork_input_8 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_blocks3bnweight_pert_pm__dual23_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_input_8_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks3bnweight_mode0__dual24_cluster_fork_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual24_cluster_fork_args =
        (_pert_blocks3bnweight_mode0__dual24_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnweight_mode0__dual24_cluster_fork, &DeeployNetwork__pert_blocks3bnweight_mode0__dual24_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_blocks3bnweight_pert_pm__dual23_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr;
} _pert_blocks3bnweight_mode0__dual24_closure_L3_args_t;

static void _pert_blocks3bnweight_mode0__dual24_closure_L3(void *_pert_blocks3bnweight_mode0__dual24_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_mode0__dual24_closure_L3_args_t *args =
      (_pert_blocks3bnweight_mode0__dual24_closure_L3_args_t *)_pert_blocks3bnweight_mode0__dual24_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_mode0__dual24_closure_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual24_closure_args =
      (_pert_blocks3bnweight_mode0__dual24_closure_args_t){.DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor =
                                                               DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr};

  // _pert_blocks3bnweight_mode0__dual24_closure CLOSURE CALL
  _pert_blocks3bnweight_mode0__dual24_closure(&DeeployNetwork__pert_blocks3bnweight_mode0__dual24_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref;
} _pert_blocks3bnbias_mode0__dual26_tiling_closure_args_t;

static void _pert_blocks3bnbias_mode0__dual26_tiling_closure(void *_pert_blocks3bnbias_mode0__dual26_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual26_tiling_closure_args_t *args =
      (_pert_blocks3bnbias_mode0__dual26_tiling_closure_args_t *)_pert_blocks3bnbias_mode0__dual26_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks3bnbias_mode0__dual26, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks3bnbias_mode0__dual26_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnbias_mode0__dual26_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnbias_mode0__dual26_chunk = (32 >> pert_blocks3bnbias_mode0__dual26_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnbias_mode0__dual26_chunk_start =
      (uint32_t)MIN(pert_blocks3bnbias_mode0__dual26_chunk * pert_blocks3bnbias_mode0__dual26_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnbias_mode0__dual26_chunk_stop =
      (uint32_t)MIN(pert_blocks3bnbias_mode0__dual26_chunk_start + pert_blocks3bnbias_mode0__dual26_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnbias_mode0__dual26_local_size = pert_blocks3bnbias_mode0__dual26_chunk_stop - pert_blocks3bnbias_mode0__dual26_chunk_start;
  uint32_t pert_blocks3bnbias_mode0__dual26_seed = ((42 + perturb_seed_base) + NUM_CORES * 19 + pert_blocks3bnbias_mode0__dual26_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnbias_mode0__dual26_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref[pert_blocks3bnbias_mode0__dual26_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref)[pert_blocks3bnbias_mode0__dual26_chunk_start],
      pert_blocks3bnbias_mode0__dual26_seed, 1u, pert_blocks3bnbias_mode0__dual26_local_size, pert_blocks3bnbias_mode0__dual26_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref[pert_blocks3bnbias_mode0__dual26_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref[32 + pert_blocks3bnbias_mode0__dual26_chunk_start],
      pert_blocks3bnbias_mode0__dual26_seed, 0u, pert_blocks3bnbias_mode0__dual26_local_size, pert_blocks3bnbias_mode0__dual26_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref;
} _pert_blocks3bnbias_mode0__dual26_cluster_fork_args_t;

static void _pert_blocks3bnbias_mode0__dual26_cluster_fork(void *_pert_blocks3bnbias_mode0__dual26_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual26_cluster_fork_args_t *args =
      (_pert_blocks3bnbias_mode0__dual26_cluster_fork_args_t *)_pert_blocks3bnbias_mode0__dual26_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_mode0__dual26_tiling_closure_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual26_tiling_closure_args =
      (_pert_blocks3bnbias_mode0__dual26_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref};

  // _pert_blocks3bnbias_mode0__dual26_tiling_closure CLOSURE CALL
  _pert_blocks3bnbias_mode0__dual26_tiling_closure(&DeeployNetwork__pert_blocks3bnbias_mode0__dual26_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr;
} _pert_blocks3bnbias_mode0__dual26_closure_args_t;

static void _pert_blocks3bnbias_mode0__dual26_closure(void *_pert_blocks3bnbias_mode0__dual26_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual26_closure_args_t *args = (_pert_blocks3bnbias_mode0__dual26_closure_args_t *)_pert_blocks3bnbias_mode0__dual26_closure_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_input_9_ref =
      (void *)((char *)DeeployNetwork_input_9 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_blocks3bnbias_pert_pm__dual25_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_input_9_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks3bnbias_mode0__dual26_cluster_fork_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual26_cluster_fork_args =
        (_pert_blocks3bnbias_mode0__dual26_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnbias_mode0__dual26_cluster_fork, &DeeployNetwork__pert_blocks3bnbias_mode0__dual26_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_blocks3bnbias_pert_pm__dual25_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr;
} _pert_blocks3bnbias_mode0__dual26_closure_L3_args_t;

static void _pert_blocks3bnbias_mode0__dual26_closure_L3(void *_pert_blocks3bnbias_mode0__dual26_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_mode0__dual26_closure_L3_args_t *args =
      (_pert_blocks3bnbias_mode0__dual26_closure_L3_args_t *)_pert_blocks3bnbias_mode0__dual26_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_mode0__dual26_closure_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual26_closure_args =
      (_pert_blocks3bnbias_mode0__dual26_closure_args_t){.DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor =
                                                             DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr};

  // _pert_blocks3bnbias_mode0__dual26_closure CLOSURE CALL
  _pert_blocks3bnbias_mode0__dual26_closure(&DeeployNetwork__pert_blocks3bnbias_mode0__dual26_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref;
} _pert_blocks4bnweight_mode0__dual31_tiling_closure_args_t;

static void _pert_blocks4bnweight_mode0__dual31_tiling_closure(void *_pert_blocks4bnweight_mode0__dual31_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual31_tiling_closure_args_t *args =
      (_pert_blocks4bnweight_mode0__dual31_tiling_closure_args_t *)_pert_blocks4bnweight_mode0__dual31_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks4bnweight_mode0__dual31, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks4bnweight_mode0__dual31_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnweight_mode0__dual31_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnweight_mode0__dual31_chunk = (32 >> pert_blocks4bnweight_mode0__dual31_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnweight_mode0__dual31_chunk_start =
      (uint32_t)MIN(pert_blocks4bnweight_mode0__dual31_chunk * pert_blocks4bnweight_mode0__dual31_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnweight_mode0__dual31_chunk_stop =
      (uint32_t)MIN(pert_blocks4bnweight_mode0__dual31_chunk_start + pert_blocks4bnweight_mode0__dual31_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnweight_mode0__dual31_local_size = pert_blocks4bnweight_mode0__dual31_chunk_stop - pert_blocks4bnweight_mode0__dual31_chunk_start;
  uint32_t pert_blocks4bnweight_mode0__dual31_seed =
      ((42 + perturb_seed_base) + NUM_CORES * 20 + pert_blocks4bnweight_mode0__dual31_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnweight_mode0__dual31_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref[pert_blocks4bnweight_mode0__dual31_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref)[pert_blocks4bnweight_mode0__dual31_chunk_start],
      pert_blocks4bnweight_mode0__dual31_seed, 1u, pert_blocks4bnweight_mode0__dual31_local_size, pert_blocks4bnweight_mode0__dual31_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref[pert_blocks4bnweight_mode0__dual31_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref[32 + pert_blocks4bnweight_mode0__dual31_chunk_start],
      pert_blocks4bnweight_mode0__dual31_seed, 0u, pert_blocks4bnweight_mode0__dual31_local_size, pert_blocks4bnweight_mode0__dual31_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref;
} _pert_blocks4bnweight_mode0__dual31_cluster_fork_args_t;

static void _pert_blocks4bnweight_mode0__dual31_cluster_fork(void *_pert_blocks4bnweight_mode0__dual31_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual31_cluster_fork_args_t *args =
      (_pert_blocks4bnweight_mode0__dual31_cluster_fork_args_t *)_pert_blocks4bnweight_mode0__dual31_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_mode0__dual31_tiling_closure_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual31_tiling_closure_args =
      (_pert_blocks4bnweight_mode0__dual31_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref};

  // _pert_blocks4bnweight_mode0__dual31_tiling_closure CLOSURE CALL
  _pert_blocks4bnweight_mode0__dual31_tiling_closure(&DeeployNetwork__pert_blocks4bnweight_mode0__dual31_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr;
} _pert_blocks4bnweight_mode0__dual31_closure_args_t;

static void _pert_blocks4bnweight_mode0__dual31_closure(void *_pert_blocks4bnweight_mode0__dual31_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual31_closure_args_t *args =
      (_pert_blocks4bnweight_mode0__dual31_closure_args_t *)_pert_blocks4bnweight_mode0__dual31_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_input_10_ref =
      (void *)((char *)DeeployNetwork_input_10 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_blocks4bnweight_pert_pm__dual30_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_input_10_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks4bnweight_mode0__dual31_cluster_fork_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual31_cluster_fork_args =
        (_pert_blocks4bnweight_mode0__dual31_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnweight_mode0__dual31_cluster_fork, &DeeployNetwork__pert_blocks4bnweight_mode0__dual31_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_blocks4bnweight_pert_pm__dual30_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr;
} _pert_blocks4bnweight_mode0__dual31_closure_L3_args_t;

static void _pert_blocks4bnweight_mode0__dual31_closure_L3(void *_pert_blocks4bnweight_mode0__dual31_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_mode0__dual31_closure_L3_args_t *args =
      (_pert_blocks4bnweight_mode0__dual31_closure_L3_args_t *)_pert_blocks4bnweight_mode0__dual31_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_mode0__dual31_closure_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual31_closure_args =
      (_pert_blocks4bnweight_mode0__dual31_closure_args_t){.DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor =
                                                               DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr};

  // _pert_blocks4bnweight_mode0__dual31_closure CLOSURE CALL
  _pert_blocks4bnweight_mode0__dual31_closure(&DeeployNetwork__pert_blocks4bnweight_mode0__dual31_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref;
} _pert_blocks4bnbias_mode0__dual33_tiling_closure_args_t;

static void _pert_blocks4bnbias_mode0__dual33_tiling_closure(void *_pert_blocks4bnbias_mode0__dual33_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual33_tiling_closure_args_t *args =
      (_pert_blocks4bnbias_mode0__dual33_tiling_closure_args_t *)_pert_blocks4bnbias_mode0__dual33_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_blocks4bnbias_mode0__dual33, Op: PerturbRademacherFixed, mode 0) -- QW exp8
  uint8_t pert_blocks4bnbias_mode0__dual33_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnbias_mode0__dual33_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnbias_mode0__dual33_chunk = (32 >> pert_blocks4bnbias_mode0__dual33_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnbias_mode0__dual33_chunk_start =
      (uint32_t)MIN(pert_blocks4bnbias_mode0__dual33_chunk * pert_blocks4bnbias_mode0__dual33_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnbias_mode0__dual33_chunk_stop =
      (uint32_t)MIN(pert_blocks4bnbias_mode0__dual33_chunk_start + pert_blocks4bnbias_mode0__dual33_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnbias_mode0__dual33_local_size = pert_blocks4bnbias_mode0__dual33_chunk_stop - pert_blocks4bnbias_mode0__dual33_chunk_start;
  uint32_t pert_blocks4bnbias_mode0__dual33_seed = ((42 + perturb_seed_base) + NUM_CORES * 21 + pert_blocks4bnbias_mode0__dual33_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnbias_mode0__dual33_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref[pert_blocks4bnbias_mode0__dual33_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref)[pert_blocks4bnbias_mode0__dual33_chunk_start],
      pert_blocks4bnbias_mode0__dual33_seed, 1u, pert_blocks4bnbias_mode0__dual33_local_size, pert_blocks4bnbias_mode0__dual33_eps);
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref[pert_blocks4bnbias_mode0__dual33_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref[32 + pert_blocks4bnbias_mode0__dual33_chunk_start],
      pert_blocks4bnbias_mode0__dual33_seed, 0u, pert_blocks4bnbias_mode0__dual33_local_size, pert_blocks4bnbias_mode0__dual33_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref;
} _pert_blocks4bnbias_mode0__dual33_cluster_fork_args_t;

static void _pert_blocks4bnbias_mode0__dual33_cluster_fork(void *_pert_blocks4bnbias_mode0__dual33_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual33_cluster_fork_args_t *args =
      (_pert_blocks4bnbias_mode0__dual33_cluster_fork_args_t *)_pert_blocks4bnbias_mode0__dual33_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_mode0__dual33_tiling_closure_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual33_tiling_closure_args =
      (_pert_blocks4bnbias_mode0__dual33_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref};

  // _pert_blocks4bnbias_mode0__dual33_tiling_closure CLOSURE CALL
  _pert_blocks4bnbias_mode0__dual33_tiling_closure(&DeeployNetwork__pert_blocks4bnbias_mode0__dual33_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr;
} _pert_blocks4bnbias_mode0__dual33_closure_args_t;

static void _pert_blocks4bnbias_mode0__dual33_closure(void *_pert_blocks4bnbias_mode0__dual33_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual33_closure_args_t *args = (_pert_blocks4bnbias_mode0__dual33_closure_args_t *)_pert_blocks4bnbias_mode0__dual33_closure_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_input_11_ref =
      (void *)((char *)DeeployNetwork_input_11 + DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_blocks4bnbias_pert_pm__dual32_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_input_11_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_blocks4bnbias_mode0__dual33_cluster_fork_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual33_cluster_fork_args =
        (_pert_blocks4bnbias_mode0__dual33_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnbias_mode0__dual33_cluster_fork, &DeeployNetwork__pert_blocks4bnbias_mode0__dual33_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_blocks4bnbias_pert_pm__dual32_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr;
} _pert_blocks4bnbias_mode0__dual33_closure_L3_args_t;

static void _pert_blocks4bnbias_mode0__dual33_closure_L3(void *_pert_blocks4bnbias_mode0__dual33_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_mode0__dual33_closure_L3_args_t *args =
      (_pert_blocks4bnbias_mode0__dual33_closure_L3_args_t *)_pert_blocks4bnbias_mode0__dual33_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_mode0__dual33_closure_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual33_closure_args =
      (_pert_blocks4bnbias_mode0__dual33_closure_args_t){.DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor =
                                                             DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr};

  // _pert_blocks4bnbias_mode0__dual33_closure CLOSURE CALL
  _pert_blocks4bnbias_mode0__dual33_closure(&DeeployNetwork__pert_blocks4bnbias_mode0__dual33_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref;
} _pert_fcweight_fp32_mode1__dual42_tiling_closure_args_t;

static void _pert_fcweight_fp32_mode1__dual42_tiling_closure(void *_pert_fcweight_fp32_mode1__dual42_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual42_tiling_closure_args_t *args =
      (_pert_fcweight_fp32_mode1__dual42_tiling_closure_args_t *)_pert_fcweight_fp32_mode1__dual42_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcweight_fp32_mode1__dual42, Op: PerturbRademacherFixed, mode 1) -- QW exp8
  uint8_t pert_fcweight_fp32_mode1__dual42_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcweight_fp32_mode1__dual42_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcweight_fp32_mode1__dual42_chunk = (288 >> pert_fcweight_fp32_mode1__dual42_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcweight_fp32_mode1__dual42_chunk_start =
      (uint32_t)MIN(pert_fcweight_fp32_mode1__dual42_chunk * pert_fcweight_fp32_mode1__dual42_core_id, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode1__dual42_chunk_stop =
      (uint32_t)MIN(pert_fcweight_fp32_mode1__dual42_chunk_start + pert_fcweight_fp32_mode1__dual42_chunk, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode1__dual42_local_size = pert_fcweight_fp32_mode1__dual42_chunk_stop - pert_fcweight_fp32_mode1__dual42_chunk_start;
  uint32_t pert_fcweight_fp32_mode1__dual42_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_fcweight_fp32_mode1__dual42_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcweight_fp32_mode1__dual42_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref[pert_fcweight_fp32_mode1__dual42_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref)[pert_fcweight_fp32_mode1__dual42_chunk_start],
      pert_fcweight_fp32_mode1__dual42_seed, 1u, pert_fcweight_fp32_mode1__dual42_local_size, pert_fcweight_fp32_mode1__dual42_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref;
} _pert_fcweight_fp32_mode1__dual42_cluster_fork_args_t;

static void _pert_fcweight_fp32_mode1__dual42_cluster_fork(void *_pert_fcweight_fp32_mode1__dual42_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual42_cluster_fork_args_t *args =
      (_pert_fcweight_fp32_mode1__dual42_cluster_fork_args_t *)_pert_fcweight_fp32_mode1__dual42_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode1__dual42_tiling_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual42_tiling_closure_args =
      (_pert_fcweight_fp32_mode1__dual42_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref};

  // _pert_fcweight_fp32_mode1__dual42_tiling_closure CLOSURE CALL
  _pert_fcweight_fp32_mode1__dual42_tiling_closure(&DeeployNetwork__pert_fcweight_fp32_mode1__dual42_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr;
} _pert_fcweight_fp32_mode1__dual42_closure_args_t;

static void _pert_fcweight_fp32_mode1__dual42_closure(void *_pert_fcweight_fp32_mode1__dual42_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual42_closure_args_t *args = (_pert_fcweight_fp32_mode1__dual42_closure_args_t *)_pert_fcweight_fp32_mode1__dual42_closure_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_input_22_ref =
      (void *)((char *)DeeployNetwork_input_22 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_fcweight_fp32_pert_p__dual41_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_input_22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcweight_fp32_mode1__dual42_cluster_fork_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual42_cluster_fork_args =
        (_pert_fcweight_fp32_mode1__dual42_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcweight_fp32_mode1__dual42_cluster_fork, &DeeployNetwork__pert_fcweight_fp32_mode1__dual42_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_fcweight_fp32_pert_p__dual41_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr;
} _pert_fcweight_fp32_mode1__dual42_closure_L3_args_t;

static void _pert_fcweight_fp32_mode1__dual42_closure_L3(void *_pert_fcweight_fp32_mode1__dual42_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode1__dual42_closure_L3_args_t *args =
      (_pert_fcweight_fp32_mode1__dual42_closure_L3_args_t *)_pert_fcweight_fp32_mode1__dual42_closure_L3_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode1__dual42_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual42_closure_args =
      (_pert_fcweight_fp32_mode1__dual42_closure_args_t){.DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor =
                                                             DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr};

  // _pert_fcweight_fp32_mode1__dual42_closure CLOSURE CALL
  _pert_fcweight_fp32_mode1__dual42_closure(&DeeployNetwork__pert_fcweight_fp32_mode1__dual42_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref;
} _pert_fcweight_fp32_mode2__dual48_tiling_closure_args_t;

static void _pert_fcweight_fp32_mode2__dual48_tiling_closure(void *_pert_fcweight_fp32_mode2__dual48_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual48_tiling_closure_args_t *args =
      (_pert_fcweight_fp32_mode2__dual48_tiling_closure_args_t *)_pert_fcweight_fp32_mode2__dual48_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacherFixed (Name: pert_fcweight_fp32_mode2__dual48, Op: PerturbRademacherFixed, mode 2) -- QW exp8
  uint8_t pert_fcweight_fp32_mode2__dual48_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcweight_fp32_mode2__dual48_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcweight_fp32_mode2__dual48_chunk = (288 >> pert_fcweight_fp32_mode2__dual48_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcweight_fp32_mode2__dual48_chunk_start =
      (uint32_t)MIN(pert_fcweight_fp32_mode2__dual48_chunk * pert_fcweight_fp32_mode2__dual48_core_id, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode2__dual48_chunk_stop =
      (uint32_t)MIN(pert_fcweight_fp32_mode2__dual48_chunk_start + pert_fcweight_fp32_mode2__dual48_chunk, (uint32_t)288);
  uint32_t pert_fcweight_fp32_mode2__dual48_local_size = pert_fcweight_fp32_mode2__dual48_chunk_stop - pert_fcweight_fp32_mode2__dual48_chunk_start;
  uint32_t pert_fcweight_fp32_mode2__dual48_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_fcweight_fp32_mode2__dual48_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcweight_fp32_mode2__dual48_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref[pert_fcweight_fp32_mode2__dual48_chunk_start],
      &((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref)[pert_fcweight_fp32_mode2__dual48_chunk_start],
      pert_fcweight_fp32_mode2__dual48_seed, 0u, pert_fcweight_fp32_mode2__dual48_local_size, pert_fcweight_fp32_mode2__dual48_eps);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref;
} _pert_fcweight_fp32_mode2__dual48_cluster_fork_args_t;

static void _pert_fcweight_fp32_mode2__dual48_cluster_fork(void *_pert_fcweight_fp32_mode2__dual48_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual48_cluster_fork_args_t *args =
      (_pert_fcweight_fp32_mode2__dual48_cluster_fork_args_t *)_pert_fcweight_fp32_mode2__dual48_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode2__dual48_tiling_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual48_tiling_closure_args =
      (_pert_fcweight_fp32_mode2__dual48_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref};

  // _pert_fcweight_fp32_mode2__dual48_tiling_closure CLOSURE CALL
  _pert_fcweight_fp32_mode2__dual48_tiling_closure(&DeeployNetwork__pert_fcweight_fp32_mode2__dual48_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr;
} _pert_fcweight_fp32_mode2__dual48_closure_args_t;

static void _pert_fcweight_fp32_mode2__dual48_closure(void *_pert_fcweight_fp32_mode2__dual48_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual48_closure_args_t *args = (_pert_fcweight_fp32_mode2__dual48_closure_args_t *)_pert_fcweight_fp32_mode2__dual48_closure_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_input_22_ref =
      (void *)((char *)DeeployNetwork_input_22 + DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_fcweight_fp32_pert_m__dual47_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_input_22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_fcweight_fp32_mode2__dual48_cluster_fork_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual48_cluster_fork_args =
        (_pert_fcweight_fp32_mode2__dual48_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_in_ref,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcweight_fp32_mode2__dual48_cluster_fork, &DeeployNetwork__pert_fcweight_fp32_mode2__dual48_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_fcweight_fp32_pert_m__dual47_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr;
} _pert_fcweight_fp32_mode2__dual48_closure_L3_args_t;

static void _pert_fcweight_fp32_mode2__dual48_closure_L3(void *_pert_fcweight_fp32_mode2__dual48_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_mode2__dual48_closure_L3_args_t *args =
      (_pert_fcweight_fp32_mode2__dual48_closure_L3_args_t *)_pert_fcweight_fp32_mode2__dual48_closure_L3_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_mode2__dual48_closure_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual48_closure_args =
      (_pert_fcweight_fp32_mode2__dual48_closure_args_t){.DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor =
                                                             DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr};

  // _pert_fcweight_fp32_mode2__dual48_closure CLOSURE CALL
  _pert_fcweight_fp32_mode2__dual48_closure(&DeeployNetwork__pert_fcweight_fp32_mode2__dual48_closure_args);

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t;

static void _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure(
    void *_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 1, 14, 700] -> [1, 14, 700, 1] (Name: wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose, Op:
  // Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 700;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src =
      (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *dst =
      (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_1;
  const uint32_t sOut_1 = sOut_2 * dimLen_3;
  const uint32_t sOut_0 = sOut_1 * dimLen_2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
      const uint32_t offS_1 = offS_0 + i_2 * sIn_2;
      const uint32_t offD_1 = offD_0 + i_2 * sOut_1;

      const uint32_t baseChunk_2 = dimLen_3 / NUM_CORES;
      const uint32_t leftover_2 = dimLen_3 - baseChunk_2 * NUM_CORES;
      const uint32_t offset_2 = baseChunk_2 * coreId + (coreId < leftover_2 ? coreId : leftover_2);
      const uint32_t chunk_2 = coreId < leftover_2 ? baseChunk_2 + 1 : baseChunk_2;
      for (uint32_t i_3 = offset_2; i_3 < offset_2 + chunk_2; i_3++) {
        const uint32_t offS_2 = offS_1 + i_3 * sIn_3;
        const uint32_t offD_2 = offD_1 + i_3 * sOut_2;

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
          const uint32_t offS_3 = offS_2 + i_1 * sIn_1;
          const uint32_t offD_3 = offD_2 + i_1 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t;

static void _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork(
    void *_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t *)
          _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args =
          (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref};

  // _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t;

static void _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure(
    void *_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t *)
          _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9800);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref =
      (void
           *)((char *)DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void
           *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed +
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_cumByteOffset
                  [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles
           [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1451592,
          .size = 9800,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args =
            (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1320520,
          .size = 9800,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t;

static void _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3(
    void *_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t *)
          _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t
      DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args =
          (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
              .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
                  DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr};

  // _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure(
      &DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_tiling_closure_args_t;

static void _wrappedInnerForwardImplConv__dual1_tiling_closure(void *_wrappedInnerForwardImplConv__dual1_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_tiling_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_tiling_closure_args_t *)_wrappedInnerForwardImplConv__dual1_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV (Name: wrappedInnerForwardImplConv__dual1, Op: RequantizedDualConv)  -- QW exp5
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImplConv__dual1_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
  uint32_t wrappedInnerForwardImplConv__dual1_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 1);
  uint32_t wrappedInnerForwardImplConv__dual1_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImplConv__dual1_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_i8_i8_i8_HWC(
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref,
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref,
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref,
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref, (int8_t *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer,
      wrappedInnerForwardImplConv__dual1_seed_w, wrappedInnerForwardImplConv__dual1_seed_b, wrappedInnerForwardImplConv__dual1_tile_xor, 15, 16,
      wrappedInnerForwardImplConv__dual1_eps_scale, NUM_CORES, 16, *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref, 14, 1,
      *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref, 14, ((16) >> 1), 4, 1, 0, 0,
      *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref,
      *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref;
} _wrappedInnerForwardImplConv__dual1_cluster_fork_args_t;

static void _wrappedInnerForwardImplConv__dual1_cluster_fork(void *_wrappedInnerForwardImplConv__dual1_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_cluster_fork_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_cluster_fork_args_t *)_wrappedInnerForwardImplConv__dual1_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv__dual1_tiling_closure_args =
      (_wrappedInnerForwardImplConv__dual1_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer =
                                                                      DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref};

  // _wrappedInnerForwardImplConv__dual1_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_tiling_closure(&DeeployNetwork__wrappedInnerForwardImplConv__dual1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_closure_args_t;

static void _wrappedInnerForwardImplConv__dual1_closure(void *_wrappedInnerForwardImplConv__dual1_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_closure_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_closure_args_t *)_wrappedInnerForwardImplConv__dual1_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
      (uint8_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
      (uint8_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 106880);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113792);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113824);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113856);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_input_12_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_17_ref =
      (void *)((char *)DeeployNetwork_input_17 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks0convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 113600);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_cmd[TILING_I],
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_size_1d[TILING_I], 700);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref) +
                475);

    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_input_12_transposed_ref);
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_17_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks0convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks0convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
        &DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
        &DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
        &DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
        &DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y[TILING_I];

    _wrappedInnerForwardImplConv__dual1_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImplConv__dual1_cluster_fork_args =
        (_wrappedInnerForwardImplConv__dual1_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer =
                                                                      DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_in_y_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_dim_im_out_y_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_left_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_padding_x_right_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_weight_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_mul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_add_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wmul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_bmul_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImplConv__dual1_cluster_fork, &DeeployNetwork__wrappedInnerForwardImplConv__dual1_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_cmd[TILING_I],
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_data_out_size_1d[TILING_I], 11216);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref) +
                7632);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr;
} _wrappedInnerForwardImplConv__dual1_closure_L3_args_t;

static void _wrappedInnerForwardImplConv__dual1_closure_L3(void *_wrappedInnerForwardImplConv__dual1_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImplConv__dual1_closure_L3_args_t *args =
      (_wrappedInnerForwardImplConv__dual1_closure_L3_args_t *)_wrappedInnerForwardImplConv__dual1_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImplConv__dual1_closure_args_t DeeployNetwork__wrappedInnerForwardImplConv__dual1_closure_args =
      (_wrappedInnerForwardImplConv__dual1_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed,
          .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr};

  // _wrappedInnerForwardImplConv__dual1_closure CLOSURE CALL
  _wrappedInnerForwardImplConv__dual1_closure(&DeeployNetwork__wrappedInnerForwardImplConv__dual1_closure_args);

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
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_args_t;

static void _bnBatchNormalization_tail_closure(void *_bnBatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_args_t *args = (_bnBatchNormalization_tail_closure_args_t *)_bnBatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
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
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_pm__dual2_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_cumByteOffset
                                                                                  [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_pm__dual4_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_cumByteOffset
                                                                                [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean__dual6_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_mean__dual6_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var__dual7_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_var__dual7_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed +
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
    mchan_transfer_1d(
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cmd[TILING_I],
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref) +
                100944);

    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_pm__dual2_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_pm__dual4_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean__dual6_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var__dual7_tensor_ref);

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
    mchan_transfer_1d(
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cmd[TILING_I],
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref) +
                12528);

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
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_L3_args_t;

static void _bnBatchNormalization_tail_closure_L3(void *_bnBatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_L3_args_t *args = (_bnBatchNormalization_tail_closure_L3_args_t *)_bnBatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = args->DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = args->DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bnBatchNormalization_tail_closure_args_t DeeployNetwork__bnBatchNormalization_tail_closure_args = (_bnBatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor,
      .DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor,
      .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

  // _bnBatchNormalization_tail_closure CLOSURE CALL
  _bnBatchNormalization_tail_closure(&DeeployNetwork__bnBatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref;
} _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure(void *_wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args_t *)_wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_1Conv__dual8, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_1Conv__dual8_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
  uint32_t wrappedInnerForwardImpl_1Conv__dual8_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 3);
  uint32_t wrappedInnerForwardImpl_1Conv__dual8_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_1Conv__dual8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref + 8,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref + ((32) >> 1), (32), 16,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer, wrappedInnerForwardImpl_1Conv__dual8_seed_w,
                                   wrappedInnerForwardImpl_1Conv__dual8_seed_b, wrappedInnerForwardImpl_1Conv__dual8_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_1Conv__dual8_eps_scale, NUM_CORES, 16, 87, 14, 8, 88, 14, ((32) >> 1), 16, 1, 0, 0, 8, 8, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref;
} _wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv__dual8_cluster_fork(void *_wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args_t *)_wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args =
      (_wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer =
                                                                        DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv__dual8_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv__dual8_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv__dual8_closure(void *_wrappedInnerForwardImpl_1Conv__dual8_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv__dual8_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv__dual8_closure_args_t *)_wrappedInnerForwardImpl_1Conv__dual8_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 69344);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88832);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88896);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88960);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2112);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_input_13_ref =
      (void *)((char *)DeeployNetwork_input_13 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_input_18_ref =
      (void *)((char *)DeeployNetwork_input_18 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_blocks1convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 41536);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1461280, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_input_13_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_input_18_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_blocks1convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_blocks1convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args =
        (_wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer =
                                                                        DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_in_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_weight_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_mul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_add_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wmul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_bmul_ref,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref =
                                                                        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv__dual8_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1350144, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv__dual8_closure_L3(void *_wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args_t *)_wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv__dual8_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_closure_args =
      (_wrappedInnerForwardImpl_1Conv__dual8_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv__dual8_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv__dual8_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_closure_args);

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
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_args_t;

static void _bn_1BatchNormalization_tail_closure(void *_bn_1BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_args_t *args = (_bn_1BatchNormalization_tail_closure_args_t *)_bn_1BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49408);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49536);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49664);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39424);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_pm__dual9_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset
                                                                                  [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_pm__dual11_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean__dual13_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_mean__dual13_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var__dual14_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_var__dual14_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed +
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
    mchan_transfer_1d(
        1481216, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_pm__dual9_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_pm__dual11_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean__dual13_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var__dual14_tensor_ref);

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
    mchan_transfer_1d(
        1320576, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

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
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_L3_args_t;

static void _bn_1BatchNormalization_tail_closure_L3(void *_bn_1BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_L3_args_t *args = (_bn_1BatchNormalization_tail_closure_L3_args_t *)_bn_1BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = args->DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = args->DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_1BatchNormalization_tail_closure_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_args = (_bn_1BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor,
      .DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

  // _bn_1BatchNormalization_tail_closure CLOSURE CALL
  _bn_1BatchNormalization_tail_closure(&DeeployNetwork__bn_1BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref;
} _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure(void *_wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args_t *)_wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_2Conv__dual15, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_2Conv__dual15_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 4);
  uint32_t wrappedInnerForwardImpl_2Conv__dual15_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 5);
  uint32_t wrappedInnerForwardImpl_2Conv__dual15_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_2Conv__dual15_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref + 16,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref + ((32) >> 1), (32), 32,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer, wrappedInnerForwardImpl_2Conv__dual15_seed_w,
                                   wrappedInnerForwardImpl_2Conv__dual15_seed_b, wrappedInnerForwardImpl_2Conv__dual15_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_2Conv__dual15_eps_scale, NUM_CORES, 16, 22, 14, 16, 23, 14, ((32) >> 1), 8, 1, 0, 0, 4, 4, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref;
} _wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_2Conv__dual15_cluster_fork(void *_wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args_t *)_wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args =
      (_wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref};

  // _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv__dual15_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv__dual15_closure_args_t;

static void _wrappedInnerForwardImpl_2Conv__dual15_closure(void *_wrappedInnerForwardImpl_2Conv__dual15_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv__dual15_closure_args_t *args =
      (_wrappedInnerForwardImpl_2Conv__dual15_closure_args_t *)_wrappedInnerForwardImpl_2Conv__dual15_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30592);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 40448);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 40512);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 40576);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 20288);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_input_14_ref =
      (void *)((char *)DeeployNetwork_input_14 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_input_19_ref =
      (void *)((char *)DeeployNetwork_input_19 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_blocks2convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_blocks2convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2112);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1451648, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_input_14_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_input_19_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_blocks2convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_blocks2convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args =
        (_wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_2Conv__dual15_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1321024, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr;
} _wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args_t;

static void _wrappedInnerForwardImpl_2Conv__dual15_closure_L3(void *_wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args_t *)_wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_2Conv__dual15_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_closure_args =
      (_wrappedInnerForwardImpl_2Conv__dual15_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr};

  // _wrappedInnerForwardImpl_2Conv__dual15_closure CLOSURE CALL
  _wrappedInnerForwardImpl_2Conv__dual15_closure(&DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_closure_args);

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
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_args_t;

static void _bn_2BatchNormalization_tail_closure(void *_bn_2BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_args_t *args = (_bn_2BatchNormalization_tail_closure_args_t *)_bn_2BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12672);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12544);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12800);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12928);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10304);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_pm__dual16_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_pm__dual18_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean__dual20_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_mean__dual20_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var__dual21_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_var__dual21_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
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
    mchan_transfer_1d(
        1452096, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_pm__dual16_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_pm__dual18_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean__dual20_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var__dual21_tensor_ref);

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
        DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

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
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_L3_args_t;

static void _bn_2BatchNormalization_tail_closure_L3(void *_bn_2BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_L3_args_t *args = (_bn_2BatchNormalization_tail_closure_L3_args_t *)_bn_2BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = args->DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = args->DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_2BatchNormalization_tail_closure_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_args = (_bn_2BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor,
      .DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

  // _bn_2BatchNormalization_tail_closure CLOSURE CALL
  _bn_2BatchNormalization_tail_closure(&DeeployNetwork__bn_2BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref;
} _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure(void *_wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args_t *)_wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_3Conv__dual22, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_3Conv__dual22_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
  uint32_t wrappedInnerForwardImpl_3Conv__dual22_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 7);
  uint32_t wrappedInnerForwardImpl_3Conv__dual22_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_3Conv__dual22_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref + 16,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref + ((64) >> 1), (64), 32,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer, wrappedInnerForwardImpl_3Conv__dual22_seed_w,
                                   wrappedInnerForwardImpl_3Conv__dual22_seed_b, wrappedInnerForwardImpl_3Conv__dual22_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_3Conv__dual22_eps_scale, NUM_CORES, 16, 5, 14, 16, 5, 8, ((64) >> 1), 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref;
} _wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual22_cluster_fork(void *_wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args_t *)_wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args =
      (_wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref};

  // _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_3Conv__dual22_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr;
} _wrappedInnerForwardImpl_3Conv__dual22_closure_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual22_closure(void *_wrappedInnerForwardImpl_3Conv__dual22_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual22_closure_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual22_closure_args_t *)_wrappedInnerForwardImpl_3Conv__dual22_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 17280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19520);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19648);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19776);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14720);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_input_15_ref =
      (void *)((char *)DeeployNetwork_input_15 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_input_20_ref =
      (void *)((char *)DeeployNetwork_input_20 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_blocks3convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_blocks3convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3712);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1444032, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_input_15_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_input_20_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_blocks3convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_blocks3convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args =
        (_wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_3Conv__dual22_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1313280, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr;
} _wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args_t;

static void _wrappedInnerForwardImpl_3Conv__dual22_closure_L3(void *_wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args_t *)_wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_3Conv__dual22_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_closure_args =
      (_wrappedInnerForwardImpl_3Conv__dual22_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr};

  // _wrappedInnerForwardImpl_3Conv__dual22_closure CLOSURE CALL
  _wrappedInnerForwardImpl_3Conv__dual22_closure(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_closure_args);

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
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_args_t;

static void _bn_3BatchNormalization_tail_closure(void *_bn_3BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_args_t *args = (_bn_3BatchNormalization_tail_closure_args_t *)_bn_3BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
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
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_pm__dual23_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset
                                                                                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_pm__dual25_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean__dual27_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_mean__dual27_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var__dual28_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_var__dual28_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
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
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_pm__dual23_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_pm__dual25_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean__dual27_tensor_ref);
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var__dual28_tensor_ref);

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
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

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
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_L3_args_t;

static void _bn_3BatchNormalization_tail_closure_L3(void *_bn_3BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_L3_args_t *args = (_bn_3BatchNormalization_tail_closure_L3_args_t *)_bn_3BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = args->DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = args->DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_3BatchNormalization_tail_closure_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_args = (_bn_3BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor,
      .DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

  // _bn_3BatchNormalization_tail_closure CLOSURE CALL
  _bn_3BatchNormalization_tail_closure(&DeeployNetwork__bn_3BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref;
} _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure(void *_wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args_t *)_wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: wrappedInnerForwardImpl_4Conv__dual29, Op: RequantizedDualConv)  -- QW exp8
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t wrappedInnerForwardImpl_4Conv__dual29_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
  uint32_t wrappedInnerForwardImpl_4Conv__dual29_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
  uint32_t wrappedInnerForwardImpl_4Conv__dual29_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_4Conv__dual29_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref + 32,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref + ((64) >> 1), (64), 64,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer, wrappedInnerForwardImpl_4Conv__dual29_seed_w,
                                   wrappedInnerForwardImpl_4Conv__dual29_seed_b, wrappedInnerForwardImpl_4Conv__dual29_tile_xor, 15, 16,
                                   wrappedInnerForwardImpl_4Conv__dual29_eps_scale, NUM_CORES, 16, 5, 8, 32, 5, 2, ((64) >> 1), 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref;
} _wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual29_cluster_fork(void *_wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args_t *)_wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args =
      (_wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref};

  // _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv__dual29_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv__dual29_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual29_closure(void *_wrappedInnerForwardImpl_4Conv__dual29_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual29_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual29_closure_args_t *)_wrappedInnerForwardImpl_4Conv__dual29_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29056);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32256);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32384);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32512);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31616);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_input_16_ref =
      (void *)((char *)DeeployNetwork_input_16 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_input_21_ref =
      (void *)((char *)DeeployNetwork_input_21 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7296);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1444352, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref);
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_input_16_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_input_21_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_blocks4convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args =
        (_wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args_t){.DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer =
                                                                         DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_weight_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_mul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_add_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_bmul_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_4Conv__dual29_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311360, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args_t;

static void _wrappedInnerForwardImpl_4Conv__dual29_closure_L3(void *_wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args_t *)_wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv__dual29_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_closure_args =
      (_wrappedInnerForwardImpl_4Conv__dual29_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
          .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
              DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr};

  // _wrappedInnerForwardImpl_4Conv__dual29_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv__dual29_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_closure_args);

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
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_args_t;

static void _bn_4BatchNormalization_closure(void *_bn_4BatchNormalization_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_args_t *args = (_bn_4BatchNormalization_closure_args_t *)_bn_4BatchNormalization_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3200);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3456);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3712);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3968);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4736);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4480);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_pm__dual30_tensor_ref =
      (void
           *)((char *)DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor +
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_pm__dual32_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean__dual34_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_mean__dual34_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_cumByteOffset
                                                                                 [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var__dual35_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_var__dual35_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_cumByteOffset
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
  int data_in_ExternalToLocal = -1;
  int running_mean_ExternalToLocal = -1;
  int saved_inv_std_LocalToExternal = -1;
  int running_var_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int bias_ExternalToLocal = -1;
  int saved_mean_LocalToExternal = -1;
  int scale_ExternalToLocal = -1;

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
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    scale_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_pm__dual30_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    bias_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_pm__dual32_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_mean_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean__dual34_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_var_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var__dual35_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (running_mean_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_mean_ExternalToLocal);
      mchan_transfer_free(running_mean_ExternalToLocal);
    }

    if (running_var_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_var_ExternalToLocal);
      mchan_transfer_free(running_var_ExternalToLocal);
    }

    if (bias_ExternalToLocal >= 0) {
      mchan_transfer_wait(bias_ExternalToLocal);
      mchan_transfer_free(bias_ExternalToLocal);
    }

    if (scale_ExternalToLocal >= 0) {
      mchan_transfer_wait(scale_ExternalToLocal);
      mchan_transfer_free(scale_ExternalToLocal);
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

    if (saved_mean_LocalToExternal >= 0) {
      mchan_transfer_wait(saved_mean_LocalToExternal);
      mchan_transfer_free(saved_mean_LocalToExternal);
    }

    if (saved_inv_std_LocalToExternal >= 0) {
      mchan_transfer_wait(saved_inv_std_LocalToExternal);
      mchan_transfer_free(saved_inv_std_LocalToExternal);
    }

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_L3_args_t;

static void _bn_4BatchNormalization_closure_L3(void *_bn_4BatchNormalization_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_L3_args_t *args = (_bn_4BatchNormalization_closure_L3_args_t *)_bn_4BatchNormalization_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = args->DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = args->DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_closure_args_t DeeployNetwork__bn_4BatchNormalization_closure_args = (_bn_4BatchNormalization_closure_args_t){
      .DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor,
      .DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor,
      .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
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
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

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
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

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
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

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
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref;

  // CLOSURE FUNCTION CALL

  // DualHalf (Name: wrappedInnerForwardImpl_5Gemm_half0__dual38, Op: DualHalf, half 0) -- QW exp8
  // contiguous per-core chunk: the strided form (c = core_id; c += NUM_CORES) lost the last iteration of cores 1..7 on
  // Siracusa (LLVM hardware-loop trip count), GAP9 was fine
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual38_chunk = (32u + NUM_CORES - 1u) / NUM_CORES;
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual38_start = MIN(wrappedInnerForwardImpl_5Gemm_half0__dual38_chunk * (uint32_t)pi_core_id(), 32u);
  uint32_t wrappedInnerForwardImpl_5Gemm_half0__dual38_stop =
      MIN(wrappedInnerForwardImpl_5Gemm_half0__dual38_start + wrappedInnerForwardImpl_5Gemm_half0__dual38_chunk, 32u);
  for (uint32_t r = 0; r < 1u; r++) {
    for (uint32_t c = wrappedInnerForwardImpl_5Gemm_half0__dual38_start; c < wrappedInnerForwardImpl_5Gemm_half0__dual38_stop; c++)
      ((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref)[r * 32u + c] =
          ((const float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref)[r * 2u * 32u + 0u * 32u + c];
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure(void *_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor = args->DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_Reshape_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_Reshape_output_0_p__dual37_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_p__dual37_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_Reshape_output_0_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_Reshape_output_0_p__dual37_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor = args->DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args_t){
          .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
          .DeeployNetwork_Reshape_output_0_p__dual37_tensor = DeeployNetwork_Reshape_output_0_p__dual37_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_args);

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
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure(void *_wrappedInnerForwardImpl_5Gemm_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor = args->DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1316);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_p__dual37_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_p__dual37_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_p__dual41_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset
                                                                                [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_p__dual43_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset
                                                                              [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_output_tensor_ref =
      (void *)((char *)DeeployNetwork_output_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset
                                                          [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);

  // Initialize DMA futures
  int B_ExternalToLocal = -1;
  int C_ExternalToLocal = -1;
  int A_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

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
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_p__dual37_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    B_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_p__dual41_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    C_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_p__dual43_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (B_ExternalToLocal >= 0) {
      mchan_transfer_wait(B_ExternalToLocal);
      mchan_transfer_free(B_ExternalToLocal);
    }

    if (C_ExternalToLocal >= 0) {
      mchan_transfer_wait(C_ExternalToLocal);
      mchan_transfer_free(C_ExternalToLocal);
    }

    if (A_ExternalToLocal >= 0) {
      mchan_transfer_wait(A_ExternalToLocal);
      mchan_transfer_free(A_ExternalToLocal);
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
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = args->DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = args->DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor = args->DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor,
      .DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor,
      .DeeployNetwork_Reshape_output_0_p__dual37_tensor = DeeployNetwork_Reshape_output_0_p__dual37_tensor,
      .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref;

  // CLOSURE FUNCTION CALL

  // DualHalf (Name: wrappedInnerForwardImpl_5Gemm_half1__dual40, Op: DualHalf, half 1) -- QW exp8
  // contiguous per-core chunk: the strided form (c = core_id; c += NUM_CORES) lost the last iteration of cores 1..7 on
  // Siracusa (LLVM hardware-loop trip count), GAP9 was fine
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual40_chunk = (32u + NUM_CORES - 1u) / NUM_CORES;
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual40_start = MIN(wrappedInnerForwardImpl_5Gemm_half1__dual40_chunk * (uint32_t)pi_core_id(), 32u);
  uint32_t wrappedInnerForwardImpl_5Gemm_half1__dual40_stop =
      MIN(wrappedInnerForwardImpl_5Gemm_half1__dual40_start + wrappedInnerForwardImpl_5Gemm_half1__dual40_chunk, 32u);
  for (uint32_t r = 0; r < 1u; r++) {
    for (uint32_t c = wrappedInnerForwardImpl_5Gemm_half1__dual40_start; c < wrappedInnerForwardImpl_5Gemm_half1__dual40_stop; c++)
      ((float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref)[r * 32u + c] =
          ((const float32_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref)[r * 2u * 32u + 1u * 32u + c];
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure(void *_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor = args->DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_Reshape_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_Reshape_output_0_m__dual39_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_m__dual39_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442048, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_Reshape_output_0_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_Reshape_output_0_m__dual39_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args;

  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor = args->DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args_t){
          .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
          .DeeployNetwork_Reshape_output_0_m__dual39_tensor = DeeployNetwork_Reshape_output_0_m__dual39_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure(void *_wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: wrappedInnerForwardImpl_5Gemm_minus__dual46, Op: Gemm)
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref;

  for (uint32_t i = 0; i < 1; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref,
        ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref,
        1, 32, 9, 0, 1, 0);
    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref +=
        1 * 32;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref +=
        32 * 9;

    ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref +=
        1 * 9;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref;
} _wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork(void *_wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref};

  // _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  float32_t *DeeployNetwork_output_minus__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure(void *_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor = args->DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  float32_t *DeeployNetwork_output_minus__dual45_tensor = args->DeeployNetwork_output_minus__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1316);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_Reshape_output_0_m__dual39_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_m__dual39_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_fcweight_fp32_pert_m__dual47_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_fcbias_fp32_pert_m__dual49_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_output_minus__dual45_tensor_ref =
      (void *)((char *)DeeployNetwork_output_minus__dual45_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr]);

  // Initialize DMA futures
  int B_ExternalToLocal = -1;
  int C_ExternalToLocal = -1;
  int A_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    A_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_Reshape_output_0_m__dual39_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    B_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_fcweight_fp32_pert_m__dual47_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    C_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_fcbias_fp32_pert_m__dual49_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (B_ExternalToLocal >= 0) {
      mchan_transfer_wait(B_ExternalToLocal);
      mchan_transfer_free(B_ExternalToLocal);
    }

    if (C_ExternalToLocal >= 0) {
      mchan_transfer_wait(C_ExternalToLocal);
      mchan_transfer_free(C_ExternalToLocal);
    }

    if (A_ExternalToLocal >= 0) {
      mchan_transfer_wait(A_ExternalToLocal);
      mchan_transfer_free(A_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_A_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_B_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_C_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_output_minus__dual45_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  float32_t *DeeployNetwork_output_minus__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = args->DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = args->DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor = args->DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  float32_t *DeeployNetwork_output_minus__dual45_tensor = args->DeeployNetwork_output_minus__dual45_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args =
      (_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args_t){
          .DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor,
          .DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor,
          .DeeployNetwork_Reshape_output_0_m__dual39_tensor = DeeployNetwork_Reshape_output_0_m__dual39_tensor,
          .DeeployNetwork_output_minus__dual45_tensor = DeeployNetwork_output_minus__dual45_tensor,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_args);

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
  int log_prob_LocalToExternal = -1;
  int labels_ExternalToLocal = -1;
  int loss_LocalToExternal = -1;
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

    if (loss_LocalToExternal >= 0) {
      mchan_transfer_wait(loss_LocalToExternal);
      mchan_transfer_free(loss_LocalToExternal);
    }

    if (log_prob_LocalToExternal >= 0) {
      mchan_transfer_wait(log_prob_LocalToExternal);
      mchan_transfer_free(log_prob_LocalToExternal);
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
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref;
} _CrossEntropyLoss_minus__dual52_tiling_closure_args_t;

static void _CrossEntropyLoss_minus__dual52_tiling_closure(void *_CrossEntropyLoss_minus__dual52_tiling_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual52_tiling_closure_args_t *args =
      (_CrossEntropyLoss_minus__dual52_tiling_closure_args_t *)_CrossEntropyLoss_minus__dual52_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref;

  // CLOSURE FUNCTION CALL

  BEGIN_SINGLE_CORE
  // SoftmaxCrossEntropyLoss (Name: CrossEntropyLoss_minus__dual52, Op: SoftmaxCrossEntropyLoss)
  float32_t sce_total_loss = 0.0f;
  for (uint32_t i = 0; i < 1; i++) {
    float32_t sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref[i * 9];
    for (uint32_t j = 1; j < 9; j++) {
      if (DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref[i * 9 + j] > sce_max_logit)
        sce_max_logit = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref[i * 9 + j];
    }
    float32_t sce_sum_exp = 0.0f;
    for (uint32_t j = 0; j < 9; j++)
      sce_sum_exp += expf(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref[i * 9 + j] - sce_max_logit);
    float32_t sce_log_sum_exp = logf(sce_sum_exp);
    for (uint32_t j = 0; j < 9; j++)
      DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref[i * 9 + j] =
          DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref[i * 9 + j] - sce_max_logit - sce_log_sum_exp;
    sce_total_loss += -(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref
                            [i * 9 + (uint32_t)(DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref[i])] -
                        sce_max_logit - sce_log_sum_exp);
  }
  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref[0] = sce_total_loss / (float32_t)1;
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref;
} _CrossEntropyLoss_minus__dual52_cluster_fork_args_t;

static void _CrossEntropyLoss_minus__dual52_cluster_fork(void *_CrossEntropyLoss_minus__dual52_cluster_fork_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual52_cluster_fork_args_t *args =
      (_CrossEntropyLoss_minus__dual52_cluster_fork_args_t *)_CrossEntropyLoss_minus__dual52_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref;
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_minus__dual52_tiling_closure_args_t DeeployNetwork__CrossEntropyLoss_minus__dual52_tiling_closure_args =
      (_CrossEntropyLoss_minus__dual52_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref};

  // _CrossEntropyLoss_minus__dual52_tiling_closure CLOSURE CALL
  _CrossEntropyLoss_minus__dual52_tiling_closure(&DeeployNetwork__CrossEntropyLoss_minus__dual52_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_minus__dual45_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual51_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr;
} _CrossEntropyLoss_minus__dual52_closure_args_t;

static void _CrossEntropyLoss_minus__dual52_closure(void *_CrossEntropyLoss_minus__dual52_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual52_closure_args_t *args = (_CrossEntropyLoss_minus__dual52_closure_args_t *)_CrossEntropyLoss_minus__dual52_closure_args;

  float32_t *DeeployNetwork_output_minus__dual45_tensor = args->DeeployNetwork_output_minus__dual45_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual51_tensor = args->DeeployNetwork_log_prob_minus__dual51_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12);
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 48);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_output_minus__dual45_tensor_ref =
      (void *)((char *)DeeployNetwork_output_minus__dual45_tensor + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_minus__dual51_tensor_ref =
      (void *)((char *)DeeployNetwork_log_prob_minus__dual51_tensor + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_cumByteOffset
                                                                          [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_output_1_ref =
      (void *)((char *)DeeployNetwork_output_1 + DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr]);

  // Initialize DMA futures
  int log_prob_LocalToExternal = -1;
  int labels_ExternalToLocal = -1;
  int loss_LocalToExternal = -1;
  int logits_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    // Transfer input tiles
    logits_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_output_minus__dual45_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    labels_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441800,
                                      .size = 8,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_input_1_ref};
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

    _CrossEntropyLoss_minus__dual52_cluster_fork_args_t DeeployNetwork__CrossEntropyLoss_minus__dual52_cluster_fork_args =
        (_CrossEntropyLoss_minus__dual52_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_logits_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_labels_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_CrossEntropyLoss_minus__dual52_cluster_fork, &DeeployNetwork__CrossEntropyLoss_minus__dual52_cluster_fork_args);

    // Transfer output tiles
    log_prob_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_log_prob_minus__dual51_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    loss_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310724,
                                      .size = 4,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_loss_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_output_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (loss_LocalToExternal >= 0) {
      mchan_transfer_wait(loss_LocalToExternal);
      mchan_transfer_free(loss_LocalToExternal);
    }

    if (log_prob_LocalToExternal >= 0) {
      mchan_transfer_wait(log_prob_LocalToExternal);
      mchan_transfer_free(log_prob_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_minus__dual45_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual51_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr;
} _CrossEntropyLoss_minus__dual52_closure_L3_args_t;

static void _CrossEntropyLoss_minus__dual52_closure_L3(void *_CrossEntropyLoss_minus__dual52_closure_L3_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_minus__dual52_closure_L3_args_t *args =
      (_CrossEntropyLoss_minus__dual52_closure_L3_args_t *)_CrossEntropyLoss_minus__dual52_closure_L3_args;

  float32_t *DeeployNetwork_output_minus__dual45_tensor = args->DeeployNetwork_output_minus__dual45_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual51_tensor = args->DeeployNetwork_log_prob_minus__dual51_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_minus__dual52_closure_args_t DeeployNetwork__CrossEntropyLoss_minus__dual52_closure_args = (_CrossEntropyLoss_minus__dual52_closure_args_t){
      .DeeployNetwork_output_minus__dual45_tensor = DeeployNetwork_output_minus__dual45_tensor,
      .DeeployNetwork_log_prob_minus__dual51_tensor = DeeployNetwork_log_prob_minus__dual51_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr};

  // _CrossEntropyLoss_minus__dual52_closure CLOSURE CALL
  _CrossEntropyLoss_minus__dual52_closure(&DeeployNetwork__CrossEntropyLoss_minus__dual52_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  float32_t *DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor;
  float32_t *DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor;
  float32_t *DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor;
  float32_t *DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor;
  float32_t *DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed;
  float32_t *DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor;
  float32_t *DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor;
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_p__dual37_tensor;
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_m__dual39_tensor;
  float32_t *DeeployNetwork_output_minus__dual45_tensor;
  float32_t *DeeployNetwork_log_prob_tensor;
  float32_t *DeeployNetwork_log_prob_minus__dual51_tensor;
  void *DeeployNetwork_pert_fcbias_fp32_mode1__dual44_scratch;
  void *DeeployNetwork_pert_fcbias_fp32_mode2__dual50_scratch;
  void *DeeployNetwork_pert_blocks0bnweight_mode0__dual3_scratch;
  void *DeeployNetwork_pert_blocks0bnbias_mode0__dual5_scratch;
  void *DeeployNetwork_pert_blocks1bnweight_mode0__dual10_scratch;
  void *DeeployNetwork_pert_blocks1bnbias_mode0__dual12_scratch;
  void *DeeployNetwork_pert_blocks2bnweight_mode0__dual17_scratch;
  void *DeeployNetwork_pert_blocks2bnbias_mode0__dual19_scratch;
  void *DeeployNetwork_pert_blocks3bnweight_mode0__dual24_scratch;
  void *DeeployNetwork_pert_blocks3bnbias_mode0__dual26_scratch;
  void *DeeployNetwork_pert_blocks4bnweight_mode0__dual31_scratch;
  void *DeeployNetwork_pert_blocks4bnbias_mode0__dual33_scratch;
  void *DeeployNetwork_pert_fcweight_fp32_mode1__dual42_scratch;
  void *DeeployNetwork_pert_fcweight_fp32_mode2__dual48_scratch;
  void *DeeployNetwork_wrappedInnerForwardImplConv__dual1_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_buffer;
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr;
    DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    _pert_fcbias_fp32_mode1__dual44_closure_L3_args_t DeeployNetwork__pert_fcbias_fp32_mode1__dual44_closure_L3_args =
        (_pert_fcbias_fp32_mode1__dual44_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor =
                                                                DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode1__dual44_tileIdxPtr};

    // _pert_fcbias_fp32_mode1__dual44_closure_L3 CLOSURE CALL
    _pert_fcbias_fp32_mode1__dual44_closure_L3(&DeeployNetwork__pert_fcbias_fp32_mode1__dual44_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr;
    DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56524);
    _pert_fcbias_fp32_mode2__dual50_closure_L3_args_t DeeployNetwork__pert_fcbias_fp32_mode2__dual50_closure_L3_args =
        (_pert_fcbias_fp32_mode2__dual50_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor =
                                                                DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_mode2__dual50_tileIdxPtr};

    // _pert_fcbias_fp32_mode2__dual50_closure_L3 CLOSURE CALL
    _pert_fcbias_fp32_mode2__dual50_closure_L3(&DeeployNetwork__pert_fcbias_fp32_mode2__dual50_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr;
    DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59028);
    _pert_blocks0bnweight_mode0__dual3_closure_L3_args_t DeeployNetwork__pert_blocks0bnweight_mode0__dual3_closure_L3_args =
        (_pert_blocks0bnweight_mode0__dual3_closure_L3_args_t){.DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor =
                                                                   DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_mode0__dual3_tileIdxPtr};

    // _pert_blocks0bnweight_mode0__dual3_closure_L3 CLOSURE CALL
    _pert_blocks0bnweight_mode0__dual3_closure_L3(&DeeployNetwork__pert_blocks0bnweight_mode0__dual3_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr;
    DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59092);
    _pert_blocks0bnbias_mode0__dual5_closure_L3_args_t DeeployNetwork__pert_blocks0bnbias_mode0__dual5_closure_L3_args =
        (_pert_blocks0bnbias_mode0__dual5_closure_L3_args_t){.DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor =
                                                                 DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor,
                                                             .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr =
                                                                 DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_mode0__dual5_tileIdxPtr};

    // _pert_blocks0bnbias_mode0__dual5_closure_L3 CLOSURE CALL
    _pert_blocks0bnbias_mode0__dual5_closure_L3(&DeeployNetwork__pert_blocks0bnbias_mode0__dual5_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr;
    DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _pert_blocks1bnweight_mode0__dual10_closure_L3_args_t DeeployNetwork__pert_blocks1bnweight_mode0__dual10_closure_L3_args =
        (_pert_blocks1bnweight_mode0__dual10_closure_L3_args_t){.DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor =
                                                                    DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_mode0__dual10_tileIdxPtr};

    // _pert_blocks1bnweight_mode0__dual10_closure_L3 CLOSURE CALL
    _pert_blocks1bnweight_mode0__dual10_closure_L3(&DeeployNetwork__pert_blocks1bnweight_mode0__dual10_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr;
    DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216852);
    _pert_blocks1bnbias_mode0__dual12_closure_L3_args_t DeeployNetwork__pert_blocks1bnbias_mode0__dual12_closure_L3_args =
        (_pert_blocks1bnbias_mode0__dual12_closure_L3_args_t){.DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor =
                                                                  DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_mode0__dual12_tileIdxPtr};

    // _pert_blocks1bnbias_mode0__dual12_closure_L3 CLOSURE CALL
    _pert_blocks1bnbias_mode0__dual12_closure_L3(&DeeployNetwork__pert_blocks1bnbias_mode0__dual12_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr;
    DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216212);
    _pert_blocks2bnweight_mode0__dual17_closure_L3_args_t DeeployNetwork__pert_blocks2bnweight_mode0__dual17_closure_L3_args =
        (_pert_blocks2bnweight_mode0__dual17_closure_L3_args_t){.DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor =
                                                                    DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_mode0__dual17_tileIdxPtr};

    // _pert_blocks2bnweight_mode0__dual17_closure_L3 CLOSURE CALL
    _pert_blocks2bnweight_mode0__dual17_closure_L3(&DeeployNetwork__pert_blocks2bnweight_mode0__dual17_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr;
    DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216980);
    _pert_blocks2bnbias_mode0__dual19_closure_L3_args_t DeeployNetwork__pert_blocks2bnbias_mode0__dual19_closure_L3_args =
        (_pert_blocks2bnbias_mode0__dual19_closure_L3_args_t){.DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor =
                                                                  DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_mode0__dual19_tileIdxPtr};

    // _pert_blocks2bnbias_mode0__dual19_closure_L3 CLOSURE CALL
    _pert_blocks2bnbias_mode0__dual19_closure_L3(&DeeployNetwork__pert_blocks2bnbias_mode0__dual19_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59156);
    _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args =
            (_wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args_t){
                .DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_input_12_transpose_tileIdxPtr};

    // _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3(&DeeployNetwork__wrappedInnerForwardImplConv__dual1_input_12_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr;
    DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216340);
    _pert_blocks3bnweight_mode0__dual24_closure_L3_args_t DeeployNetwork__pert_blocks3bnweight_mode0__dual24_closure_L3_args =
        (_pert_blocks3bnweight_mode0__dual24_closure_L3_args_t){.DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor =
                                                                    DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_mode0__dual24_tileIdxPtr};

    // _pert_blocks3bnweight_mode0__dual24_closure_L3 CLOSURE CALL
    _pert_blocks3bnweight_mode0__dual24_closure_L3(&DeeployNetwork__pert_blocks3bnweight_mode0__dual24_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr;
    DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217108);
    _pert_blocks3bnbias_mode0__dual26_closure_L3_args_t DeeployNetwork__pert_blocks3bnbias_mode0__dual26_closure_L3_args =
        (_pert_blocks3bnbias_mode0__dual26_closure_L3_args_t){.DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor =
                                                                  DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_mode0__dual26_tileIdxPtr};

    // _pert_blocks3bnbias_mode0__dual26_closure_L3 CLOSURE CALL
    _pert_blocks3bnbias_mode0__dual26_closure_L3(&DeeployNetwork__pert_blocks3bnbias_mode0__dual26_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr;
    DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 216596);
    _pert_blocks4bnweight_mode0__dual31_closure_L3_args_t DeeployNetwork__pert_blocks4bnweight_mode0__dual31_closure_L3_args =
        (_pert_blocks4bnweight_mode0__dual31_closure_L3_args_t){.DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor =
                                                                    DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_mode0__dual31_tileIdxPtr};

    // _pert_blocks4bnweight_mode0__dual31_closure_L3 CLOSURE CALL
    _pert_blocks4bnweight_mode0__dual31_closure_L3(&DeeployNetwork__pert_blocks4bnweight_mode0__dual31_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr;
    DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217364);
    _pert_blocks4bnbias_mode0__dual33_closure_L3_args_t DeeployNetwork__pert_blocks4bnbias_mode0__dual33_closure_L3_args =
        (_pert_blocks4bnbias_mode0__dual33_closure_L3_args_t){.DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor =
                                                                  DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_mode0__dual33_tileIdxPtr};

    // _pert_blocks4bnbias_mode0__dual33_closure_L3 CLOSURE CALL
    _pert_blocks4bnbias_mode0__dual33_closure_L3(&DeeployNetwork__pert_blocks4bnbias_mode0__dual33_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr;
    DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56596);
    _pert_fcweight_fp32_mode1__dual42_closure_L3_args_t DeeployNetwork__pert_fcweight_fp32_mode1__dual42_closure_L3_args =
        (_pert_fcweight_fp32_mode1__dual42_closure_L3_args_t){.DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor =
                                                                  DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode1__dual42_tileIdxPtr};

    // _pert_fcweight_fp32_mode1__dual42_closure_L3 CLOSURE CALL
    _pert_fcweight_fp32_mode1__dual42_closure_L3(&DeeployNetwork__pert_fcweight_fp32_mode1__dual42_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr;
    DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 57748);
    _pert_fcweight_fp32_mode2__dual48_closure_L3_args_t DeeployNetwork__pert_fcweight_fp32_mode2__dual48_closure_L3_args =
        (_pert_fcweight_fp32_mode2__dual48_closure_L3_args_t){.DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor =
                                                                  DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_mode2__dual48_tileIdxPtr};

    // _pert_fcweight_fp32_mode2__dual48_closure_L3 CLOSURE CALL
    _pert_fcweight_fp32_mode2__dual48_closure_L3(&DeeployNetwork__pert_fcweight_fp32_mode2__dual48_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;
    DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59188);
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args =
        (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t){
            .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr};

    // _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3 CLOSURE CALL
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3(&DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217620);
    _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args =
            (_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
                .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
                    DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr};

    // _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3(
        &DeeployNetwork__wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59188);
    _wrappedInnerForwardImplConv__dual1_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImplConv__dual1_closure_L3_args =
        (_wrappedInnerForwardImplConv__dual1_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed = DeeployNetwork_wrappedInnerForwardImplConv__dual1_input_12_transposed,
            .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImplConv__dual1_tileIdxPtr};

    // _wrappedInnerForwardImplConv__dual1_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImplConv__dual1_closure_L3(&DeeployNetwork__wrappedInnerForwardImplConv__dual1_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 217620);
    _bnBatchNormalization_tail_closure_L3_args_t DeeployNetwork__bnBatchNormalization_tail_closure_L3_args = (_bnBatchNormalization_tail_closure_L3_args_t){
        .DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor = DeeployNetwork_blocks0bnweight_pert_pm__dual2_tensor,
        .DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor = DeeployNetwork_blocks0bnbias_pert_pm__dual4_tensor,
        .DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork_wrappedInnerForwardImplConv__dual1_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

    // _bnBatchNormalization_tail_closure_L3 CLOSURE CALL
    _bnBatchNormalization_tail_closure_L3(&DeeployNetwork__bnBatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59028);
    _wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args =
        (_wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv__dual8_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv__dual8_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv__dual8_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_1Conv__dual8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 98452);
    _bn_1BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args =
        (_bn_1BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor = DeeployNetwork_blocks1bnweight_pert_pm__dual9_tensor,
            .DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor = DeeployNetwork_blocks1bnbias_pert_pm__dual11_tensor,
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_1Conv__dual8_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

    // _bn_1BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_1BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args =
        (_wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_2Conv__dual15_tileIdxPtr};

    // _wrappedInnerForwardImpl_2Conv__dual15_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_2Conv__dual15_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_2Conv__dual15_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 69204);
    _bn_2BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args =
        (_bn_2BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor = DeeployNetwork_blocks2bnweight_pert_pm__dual16_tensor,
            .DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor = DeeployNetwork_blocks2bnbias_pert_pm__dual18_tensor,
            .DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_2Conv__dual15_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

    // _bn_2BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_2BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    _wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args =
        (_wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_3Conv__dual22_tileIdxPtr};

    // _wrappedInnerForwardImpl_3Conv__dual22_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_3Conv__dual22_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_3Conv__dual22_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61460);
    _bn_3BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args =
        (_bn_3BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor = DeeployNetwork_blocks3bnweight_pert_pm__dual23_tensor,
            .DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor = DeeployNetwork_blocks3bnbias_pert_pm__dual25_tensor,
            .DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_3Conv__dual22_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

    // _bn_3BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_3BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 64020);
    _wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args =
        (_wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv__dual29_tileIdxPtr};

    // _wrappedInnerForwardImpl_4Conv__dual29_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_4Conv__dual29_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_4Conv__dual29_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
    DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58900);
    DeeployNetwork_bn_4BatchNormalization_sm_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61972);
    DeeployNetwork_bn_4BatchNormalization_siv_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61716);
    _bn_4BatchNormalization_closure_L3_args_t DeeployNetwork__bn_4BatchNormalization_closure_L3_args = (_bn_4BatchNormalization_closure_L3_args_t){
        .DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor = DeeployNetwork_blocks4bnweight_pert_pm__dual30_tensor,
        .DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor = DeeployNetwork_blocks4bnbias_pert_pm__dual32_tensor,
        .DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork_wrappedInnerForwardImpl_4Conv__dual29_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
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

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr;
    DeeployNetwork_Reshape_output_0_p__dual37_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59192);
    _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args_t){
            .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
            .DeeployNetwork_Reshape_output_0_p__dual37_tensor = DeeployNetwork_Reshape_output_0_p__dual37_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half0__dual38_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half0__dual38_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
    DeeployNetwork_output_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59156);
    _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor = DeeployNetwork_fcbias_fp32_pert_p__dual43_tensor,
                                                           .DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor =
                                                               DeeployNetwork_fcweight_fp32_pert_p__dual41_tensor,
                                                           .DeeployNetwork_Reshape_output_0_p__dual37_tensor = DeeployNetwork_Reshape_output_0_p__dual37_tensor,
                                                           .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr;
    DeeployNetwork_Reshape_output_0_m__dual39_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args_t){
            .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
            .DeeployNetwork_Reshape_output_0_m__dual39_tensor = DeeployNetwork_Reshape_output_0_m__dual39_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_half1__dual40_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_half1__dual40_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr;
    DeeployNetwork_output_minus__dual45_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args_t){
            .DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor = DeeployNetwork_fcbias_fp32_pert_m__dual49_tensor,
            .DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor = DeeployNetwork_fcweight_fp32_pert_m__dual47_tensor,
            .DeeployNetwork_Reshape_output_0_m__dual39_tensor = DeeployNetwork_Reshape_output_0_m__dual39_tensor,
            .DeeployNetwork_output_minus__dual45_tensor = DeeployNetwork_output_minus__dual45_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_minus__dual46_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_minus__dual46_closure_L3_args);
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

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr;
    DeeployNetwork_log_prob_minus__dual51_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56532);
    _CrossEntropyLoss_minus__dual52_closure_L3_args_t DeeployNetwork__CrossEntropyLoss_minus__dual52_closure_L3_args =
        (_CrossEntropyLoss_minus__dual52_closure_L3_args_t){.DeeployNetwork_output_minus__dual45_tensor = DeeployNetwork_output_minus__dual45_tensor,
                                                            .DeeployNetwork_log_prob_minus__dual51_tensor = DeeployNetwork_log_prob_minus__dual51_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_minus__dual52_tileIdxPtr};

    // _CrossEntropyLoss_minus__dual52_closure_L3 CLOSURE CALL
    _CrossEntropyLoss_minus__dual52_closure_L3(&DeeployNetwork__CrossEntropyLoss_minus__dual52_closure_L3_args);
  }
}

void InitTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 113888);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 113888));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 237108);

  DeeployNetwork_input_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56516);
  DeeployNetwork_input_2 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56388);
  DeeployNetwork_input_3 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56420);
  DeeployNetwork_input_4 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55968);
  DeeployNetwork_input_5 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56032);
  DeeployNetwork_input_6 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56096);
  DeeployNetwork_input_7 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56160);
  DeeployNetwork_input_8 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55200);
  DeeployNetwork_input_9 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55328);
  DeeployNetwork_input_10 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55456);
  DeeployNetwork_input_11 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55584);
  DeeployNetwork_input_12 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56452);
  DeeployNetwork_input_13 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 49952);
  DeeployNetwork_input_14 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 52000);
  DeeployNetwork_input_15 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 46368);
  DeeployNetwork_input_16 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39200);
  DeeployNetwork_input_17 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56484);
  DeeployNetwork_input_18 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56224);
  DeeployNetwork_input_19 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56288);
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
