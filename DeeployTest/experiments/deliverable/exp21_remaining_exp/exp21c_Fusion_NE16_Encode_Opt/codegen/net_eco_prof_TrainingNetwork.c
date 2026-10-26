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

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {16715, 28051, 19407, 24750, 20593, 26069, 21962, 15816};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {3887854, 4028793, 3998692, 4560022, 4527256, 4318277, 4195455, 4117846,
                                                                              3644722, 4679690, 3915633, 4146882, 4516702, 3937397, 4507614, 4631827};

static PI_L2 int32_t DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor[16] = {5779922, 6811814, 7569997, 7017406, 6391308, 6797053, 6348470, 6703570,
                                                                              7695533, 4777151, 6888298, 6076617, 7800891, 6162310, 4713482, 8093983};

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor[8] = {85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor[32] = {
    11216862, 10953840, 8883199, 8006243, 9415140, 11637480, 7831187,  5033047, 8487970,  8172161,  8744505,  12902138, 8150436,  8865462, 7886326,  11680739,
    9828662,  10359709, 9262086, 8927959, 7369362, 7468230,  10595193, 7107887, 12231511, 11419024, 10140785, 6934845,  11424804, 5359367, 12871468, 10140571};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor[32] = {15255979, 18601466, 16480003, 16604952, 16636081, 17501801, 18550852, 19833580,
                                                                              22784586, 15600707, 20171281, 19515017, 18112376, 18597217, 11157841, 21481128,
                                                                              18522609, 19594714, 17518029, 20954163, 17792839, 17273134, 20128746, 16975140,
                                                                              23468860, 19124083, 19414542, 16972130, 22860964, 15612485, 16032134, 19108972};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor[16] = {234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192,
                                                                              219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks2convweight_int8_pmul_tensor[16] = {187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670,
                                                                              249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {421, 251, 363, 284, 342, 270, 321, 445};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_mean_tensor[8] = {-0.0019563890527933836f,  -4.265925963409245e-05f,  0.0010741822188720107f,
                                                                         -0.00022483340580947697f, -3.8470159779535607e-05f, -0.00020210926595609635f,
                                                                         0.00030828602029941976f,  0.00018801260739564896f};

static PI_L2 float32_t DeeployNetwork_blocks0bnrunning_var_tensor[8] = {10794.6962890625f, 1960.42578125f,     1352.2359619140625f, 1431.591064453125f,
                                                                        4565.24951171875f, 1289.480712890625f, 1300.5797119140625f, 14838.5048828125f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {69, 67, 67, 59, 60, 62, 64, 66,
                                                                                                                 74, 58, 69, 65, 60, 69, 60, 58};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_mean_tensor[16] = {
    -0.2444174885749817f, 0.21689638495445251f, 0.14622187614440918f, -1.0542712211608887f, -1.0498594045639038f, -0.22530169785022736f,
    -1.1160809993743896f, 1.0315797328948975f,  1.0182175636291504f,  0.12470299005508423f, -1.3734524250030518f, -0.7460818886756897f,
    -0.670293390750885f,  -1.0082124471664429f, 0.6750643253326416f,  0.8555498123168945f};

static PI_L2 float32_t DeeployNetwork_blocks1bnrunning_var_tensor[16] = {1.4679807424545288f, 1.4829299449920654f, 4.743453502655029f,  5.358211040496826f,
                                                                         4.712860107421875f,  4.743685245513916f,  4.386803150177002f,  2.8539552688598633f,
                                                                         3.485712766647339f,  1.066118597984314f,  10.345640182495117f, 5.888366222381592f,
                                                                         1.6831457614898682f, 3.925990581512451f,  2.867671489715576f,  2.335334300994873f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {136, 115, 104, 112, 123, 116, 124, 117,
                                                                                                                 102, 165, 114, 129, 101, 128, 167, 97};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_mean_tensor[16] = {
    -0.4543471932411194f,  0.033739298582077026f, -0.16740328073501587f, -0.633358359336853f,    0.06192547827959061f, -0.29306620359420776f,
    -0.18620505928993225f, -0.16867128014564514f, 0.24714651703834534f,  -0.09643757343292236f,  -0.2440318465232849f, -0.32666322588920593f,
    0.3093155026435852f,   -0.48663511872291565f, -0.43303781747817993f, -0.0018389797769486904f};

static PI_L2 float32_t DeeployNetwork_blocks2bnrunning_var_tensor[16] = {0.6995296478271484f, 0.35785973072052f,   1.5051546096801758f,  0.8668981790542603f,
                                                                         0.4478089213371277f, 0.4923137128353119f, 1.185558557510376f,   0.9423964023590088f,
                                                                         1.2930549383163452f, 0.180169016122818f,  0.5536366105079651f,  0.6328718662261963f,
                                                                         0.5637519955635071f, 0.537251353263855f,  0.16963136196136475f, 0.5420557260513306f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    82, 84, 104, 115, 98, 79, 118, 183, 109, 113, 105, 71, 113, 104, 117, 79, 94, 89, 99, 103, 125, 123, 87, 130, 75, 81, 91, 133, 81, 172, 72, 91};

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_mean_tensor[32] = {
    -0.6443479657173157f, -0.8780460357666016f, -0.9605752229690552f, -0.728754460811615f,  -1.379516839981079f,   0.36568009853363037f, -0.976334273815155f,
    -0.8195656538009644f, -0.3788541853427887f, -1.4433526992797852f, -1.4827815294265747f, -0.8990912437438965f,  -0.4861598014831543f, -1.4708205461502075f,
    -0.7284005284309387f, 0.6005582213401794f,  1.5618020296096802f,  -1.0277910232543945f, -1.1303950548171997f,  -1.396936058998108f,  -1.281982183456421f,
    -0.4754037857055664f, -1.0184197425842285f, -1.0318992137908936f, -0.5048392415046692f, -0.43774089217185974f, -1.4235376119613647f, -0.006140610668808222f,
    -0.9846276044845581f, -0.4856228530406952f, -0.5889110565185547f, -1.107672095298767f};

static PI_L2 float32_t DeeployNetwork_blocks3bnrunning_var_tensor[32] = {
    0.2816118896007538f,  0.3946440517902374f,  0.4444875419139862f,  0.5381677150726318f,  0.43645036220550537f, 0.7069288492202759f,  0.7624120116233826f,
    0.35157787799835205f, 0.452341765165329f,   0.3499625027179718f,  0.6326277852058411f,  0.3337295353412628f,  0.26218539476394653f, 0.6095684766769409f,
    0.5870797634124756f,  0.5308420062065125f,  0.541131854057312f,   0.35709550976753235f, 0.3786107897758484f,  0.33237600326538086f, 0.607239305973053f,
    0.304384708404541f,   0.47302573919296265f, 0.47411835193634033f, 0.31873422861099243f, 0.40998321771621704f, 0.5107912421226501f,  0.35335618257522583f,
    0.2936612665653229f,  0.4064280688762665f,  0.3643214702606201f,  0.36573007702827454f};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    163, 133, 150, 149, 149, 142, 134, 125, 109, 159, 123, 127, 137, 133, 222, 115,
    134, 127, 142, 118, 139, 144, 123, 146, 106, 130, 128, 146, 108, 159, 155, 130};

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_mean_tensor[32] = {
    0.42469096183776855f,  0.32596051692962646f, 0.10398848354816437f,  -0.3451862037181854f,  -0.36299625039100647f, -0.19927622377872467f,
    -0.446662962436676f,   -0.6463051438331604f, -0.5271444320678711f,  -0.20434394478797913f, 0.4471416473388672f,   -0.15140807628631592f,
    -0.07134925574064255f, 0.40762627124786377f, -0.06986096501350403f, 0.3484388291835785f,   -0.0795891210436821f,  -0.2456866353750229f,
    0.1434946060180664f,   0.034666508436203f,   0.5462859272956848f,   0.46823567152023315f,  -0.3238901197910309f,  -0.6293811798095703f,
    0.325870156288147f,    0.07531605660915375f, 0.19413888454437256f,  -0.15748722851276398f, 0.16247861087322235f,  -0.5395596623420715f,
    0.23268234729766846f,  -0.3300189971923828f};

static PI_L2 float32_t DeeployNetwork_blocks4bnrunning_var_tensor[32] = {
    0.1964213103055954f,  0.17257089912891388f, 0.14762769639492035f, 0.3156340420246124f,  0.2242252230644226f,  0.2599714398384094f,  0.13144634664058685f,
    0.21899321675300598f, 0.2966296672821045f,  0.2763214111328125f,  0.21101363003253937f, 0.1880793422460556f,  0.27006587386131287f, 0.1884487271308899f,
    0.19838258624076843f, 0.16746504604816437f, 0.2212221771478653f,  0.17600755393505096f, 0.19948957860469818f, 0.1521081030368805f,  0.17710629105567932f,
    0.2113693207502365f,  0.3071841895580292f,  0.31239357590675354f, 0.23397785425186157f, 0.19457979500293732f, 0.09693852812051773f, 0.241731658577919f,
    0.3287730813026428f,  0.19810181856155396f, 0.1785918027162552f,  0.23535661399364471f};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_cumByteOffset[1] = {0};

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

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[24];
void *DeeployNetwork_outputs[2];
extern struct pi_device cluster_dev;
typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks0convbias_rqsadd_tiling_closure(void *_rqsp_blocks0convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks0convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks0convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks0convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks0convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks0convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks0convbias_rqsadd_chunk = (8 >> rqsp_blocks0convbias_rqsadd_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks0convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks0convbias_rqsadd_chunk * rqsp_blocks0convbias_rqsadd_core_id, (uint32_t)8);
  uint32_t rqsp_blocks0convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks0convbias_rqsadd_chunk_start + rqsp_blocks0convbias_rqsadd_chunk, (uint32_t)8);
  uint32_t rqsp_blocks0convbias_rqsadd_local_size = rqsp_blocks0convbias_rqsadd_chunk_stop - rqsp_blocks0convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks0convbias_rqsadd_channel_start_offset = rqsp_blocks0convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 1 + rqsp_blocks0convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks0convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks0convbias_rqsadd_local_size,
        rqsp_blocks0convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[1] + rqsp_blocks0convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks0convbias_rqsadd_local_size,
        rqsp_blocks0convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks0convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks0convbias_rqsadd_cluster_fork(void *_rqsp_blocks0convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks0convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks0convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks0convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref};

  // _rqsp_blocks0convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks0convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks0convbias_rqsadd_closure_args_t;

static void _rqsp_blocks0convbias_rqsadd_closure(void *_rqsp_blocks0convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_closure_args_t *args = (_rqsp_blocks0convbias_rqsadd_closure_args_t *)_rqsp_blocks0convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_input_17_ref =
      (void *)((char *)DeeployNetwork_input_17 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);

  const static char rqsp_blocks0convbias_rqsadd_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks0convbias_rqsadd_L2_prefix[] = "[rqsp_blocks0convbias_rqsadd_L2][SB][8 ops][Tile ";

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks0convbias_rqsadd_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_input_17_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks0convbias_rqsadd_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks0convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks0convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks0convbias_rqsadd_cluster_fork_args);

    rqsp_blocks0convbias_rqsadd_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks0convbias_rqsadd_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_measurement = rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks0convbias_rqsadd_L2_kernel_measurement =
        rqsp_blocks0convbias_rqsadd_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks0convbias_rqsadd_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_measurement = rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks0convbias_rqsadd_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_measurement,
           rqsp_blocks0convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks0convbias_rqsadd_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks0convbias_rqsadd_L2_kernel_measurement,
           rqsp_blocks0convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks0convbias_rqsadd_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_measurement,
           rqsp_blocks0convbias_rqsadd_L2_suffix);

    uint32_t total = rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks0convbias_rqsadd_L2_kernel_measurement +
                     rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks0convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks0convbias_rqsadd_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks0convbias_rqsadd_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks0convbias_rqsadd_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks0convbias_rqsadd_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks0convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks0convbias_rqsadd_closure_L3(void *_rqsp_blocks0convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks0convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks0convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_args = (_rqsp_blocks0convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks0convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks0convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref;
} _pert_blocks0bnweight_tiling_closure_args_t;

static void _pert_blocks0bnweight_tiling_closure(void *_pert_blocks0bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_tiling_closure_args_t *args = (_pert_blocks0bnweight_tiling_closure_args_t *)_pert_blocks0bnweight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks0bnweight, Op: PerturbRademacher)
  uint8_t pert_blocks0bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnweight_chunk = (8 >> pert_blocks0bnweight_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnweight_chunk_start = (uint32_t)MIN(pert_blocks0bnweight_chunk * pert_blocks0bnweight_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnweight_chunk_stop = (uint32_t)MIN(pert_blocks0bnweight_chunk_start + pert_blocks0bnweight_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnweight_local_size = pert_blocks0bnweight_chunk_stop - pert_blocks0bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 12 + pert_blocks0bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref[pert_blocks0bnweight_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref[pert_blocks0bnweight_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks0bnweight_local_size, zo_eco_mom[12] + pert_blocks0bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref[pert_blocks0bnweight_chunk_start],
        (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref[pert_blocks0bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_blocks0bnweight_local_size, pert_blocks0bnweight_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref[pert_blocks0bnweight_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref[pert_blocks0bnweight_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks0bnweight_local_size, pert_blocks0bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref;
} _pert_blocks0bnweight_cluster_fork_args_t;

static void _pert_blocks0bnweight_cluster_fork(void *_pert_blocks0bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_cluster_fork_args_t *args = (_pert_blocks0bnweight_cluster_fork_args_t *)_pert_blocks0bnweight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_tiling_closure_args_t DeeployNetwork__pert_blocks0bnweight_tiling_closure_args = (_pert_blocks0bnweight_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref};

  // _pert_blocks0bnweight_tiling_closure CLOSURE CALL
  _pert_blocks0bnweight_tiling_closure(&DeeployNetwork__pert_blocks0bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr;
} _pert_blocks0bnweight_closure_args_t;

static void _pert_blocks0bnweight_closure(void *_pert_blocks0bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_closure_args_t *args = (_pert_blocks0bnweight_closure_args_t *)_pert_blocks0bnweight_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor = args->DeeployNetwork_blocks0bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_blocks0bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr]);

  const static char pert_blocks0bnweight_L2_suffix[] = " cycles \n";

  const static char pert_blocks0bnweight_L2_prefix[] = "[pert_blocks0bnweight_L2][SB][8 ops][Tile ";

  static uint32_t pert_blocks0bnweight_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks0bnweight_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks0bnweight_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks0bnweight_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks0bnweight_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks0bnweight_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks0bnweight_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_input_2_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks0bnweight_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks0bnweight_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks0bnweight_cluster_fork_args_t DeeployNetwork__pert_blocks0bnweight_cluster_fork_args = (_pert_blocks0bnweight_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnweight_cluster_fork, &DeeployNetwork__pert_blocks0bnweight_cluster_fork_args);

    pert_blocks0bnweight_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks0bnweight_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_blocks0bnweight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks0bnweight_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks0bnweight_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks0bnweight_L2_ingress_dma_wait_measurement =
        pert_blocks0bnweight_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks0bnweight_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks0bnweight_L2_kernel_measurement =
        pert_blocks0bnweight_L2_kernel_end_measurements[PROFILING_I] - pert_blocks0bnweight_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks0bnweight_L2_egress_dma_wait_measurement =
        pert_blocks0bnweight_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks0bnweight_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks0bnweight_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks0bnweight_L2_ingress_dma_wait_measurement,
           pert_blocks0bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks0bnweight_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks0bnweight_L2_kernel_measurement,
           pert_blocks0bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks0bnweight_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks0bnweight_L2_egress_dma_wait_measurement,
           pert_blocks0bnweight_L2_suffix);

    uint32_t total =
        pert_blocks0bnweight_L2_ingress_dma_wait_measurement + pert_blocks0bnweight_L2_kernel_measurement + pert_blocks0bnweight_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks0bnweight_L2_ingress_dma_wait_measurement + pert_blocks0bnweight_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks0bnweight_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks0bnweight_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks0bnweight_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr;
} _pert_blocks0bnweight_closure_L3_args_t;

static void _pert_blocks0bnweight_closure_L3(void *_pert_blocks0bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnweight_closure_L3_args_t *args = (_pert_blocks0bnweight_closure_L3_args_t *)_pert_blocks0bnweight_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor = args->DeeployNetwork_blocks0bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnweight_closure_args_t DeeployNetwork__pert_blocks0bnweight_closure_args = (_pert_blocks0bnweight_closure_args_t){
      .DeeployNetwork_blocks0bnweight_pert_tensor = DeeployNetwork_blocks0bnweight_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr};

  // _pert_blocks0bnweight_closure CLOSURE CALL
  _pert_blocks0bnweight_closure(&DeeployNetwork__pert_blocks0bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref;
} _pert_blocks0bnbias_tiling_closure_args_t;

static void _pert_blocks0bnbias_tiling_closure(void *_pert_blocks0bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_tiling_closure_args_t *args = (_pert_blocks0bnbias_tiling_closure_args_t *)_pert_blocks0bnbias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks0bnbias, Op: PerturbRademacher)
  uint8_t pert_blocks0bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks0bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks0bnbias_chunk = (8 >> pert_blocks0bnbias_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks0bnbias_chunk_start = (uint32_t)MIN(pert_blocks0bnbias_chunk * pert_blocks0bnbias_core_id, (uint32_t)8);
  uint32_t pert_blocks0bnbias_chunk_stop = (uint32_t)MIN(pert_blocks0bnbias_chunk_start + pert_blocks0bnbias_chunk, (uint32_t)8);
  uint32_t pert_blocks0bnbias_local_size = pert_blocks0bnbias_chunk_stop - pert_blocks0bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 13 + pert_blocks0bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks0bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref[pert_blocks0bnbias_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref[pert_blocks0bnbias_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks0bnbias_local_size, zo_eco_mom[13] + pert_blocks0bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref[pert_blocks0bnbias_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref[pert_blocks0bnbias_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_blocks0bnbias_local_size, pert_blocks0bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref[pert_blocks0bnbias_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref[pert_blocks0bnbias_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks0bnbias_local_size, pert_blocks0bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref;
} _pert_blocks0bnbias_cluster_fork_args_t;

static void _pert_blocks0bnbias_cluster_fork(void *_pert_blocks0bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_cluster_fork_args_t *args = (_pert_blocks0bnbias_cluster_fork_args_t *)_pert_blocks0bnbias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_tiling_closure_args_t DeeployNetwork__pert_blocks0bnbias_tiling_closure_args = (_pert_blocks0bnbias_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref};

  // _pert_blocks0bnbias_tiling_closure CLOSURE CALL
  _pert_blocks0bnbias_tiling_closure(&DeeployNetwork__pert_blocks0bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr;
} _pert_blocks0bnbias_closure_args_t;

static void _pert_blocks0bnbias_closure(void *_pert_blocks0bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_closure_args_t *args = (_pert_blocks0bnbias_closure_args_t *)_pert_blocks0bnbias_closure_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor = args->DeeployNetwork_blocks0bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_blocks0bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr]);

  const static char pert_blocks0bnbias_L2_suffix[] = " cycles \n";

  const static char pert_blocks0bnbias_L2_prefix[] = "[pert_blocks0bnbias_L2][SB][8 ops][Tile ";

  static uint32_t pert_blocks0bnbias_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks0bnbias_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks0bnbias_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks0bnbias_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks0bnbias_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks0bnbias_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks0bnbias_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks0bnbias_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks0bnbias_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks0bnbias_cluster_fork_args_t DeeployNetwork__pert_blocks0bnbias_cluster_fork_args = (_pert_blocks0bnbias_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks0bnbias_cluster_fork, &DeeployNetwork__pert_blocks0bnbias_cluster_fork_args);

    pert_blocks0bnbias_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks0bnbias_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_blocks0bnbias_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks0bnbias_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks0bnbias_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks0bnbias_L2_ingress_dma_wait_measurement =
        pert_blocks0bnbias_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks0bnbias_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks0bnbias_L2_kernel_measurement =
        pert_blocks0bnbias_L2_kernel_end_measurements[PROFILING_I] - pert_blocks0bnbias_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks0bnbias_L2_egress_dma_wait_measurement =
        pert_blocks0bnbias_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks0bnbias_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks0bnbias_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks0bnbias_L2_ingress_dma_wait_measurement,
           pert_blocks0bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks0bnbias_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks0bnbias_L2_kernel_measurement, pert_blocks0bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks0bnbias_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks0bnbias_L2_egress_dma_wait_measurement,
           pert_blocks0bnbias_L2_suffix);

    uint32_t total =
        pert_blocks0bnbias_L2_ingress_dma_wait_measurement + pert_blocks0bnbias_L2_kernel_measurement + pert_blocks0bnbias_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks0bnbias_L2_ingress_dma_wait_measurement + pert_blocks0bnbias_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks0bnbias_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks0bnbias_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks0bnbias_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr;
} _pert_blocks0bnbias_closure_L3_args_t;

static void _pert_blocks0bnbias_closure_L3(void *_pert_blocks0bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks0bnbias_closure_L3_args_t *args = (_pert_blocks0bnbias_closure_L3_args_t *)_pert_blocks0bnbias_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor = args->DeeployNetwork_blocks0bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks0bnbias_closure_args_t DeeployNetwork__pert_blocks0bnbias_closure_args = (_pert_blocks0bnbias_closure_args_t){
      .DeeployNetwork_blocks0bnbias_pert_tensor = DeeployNetwork_blocks0bnbias_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr};

  // _pert_blocks0bnbias_closure CLOSURE CALL
  _pert_blocks0bnbias_closure(&DeeployNetwork__pert_blocks0bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref;
} _pert_fcbias_fp32_tiling_closure_args_t;

static void _pert_fcbias_fp32_tiling_closure(void *_pert_fcbias_fp32_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_tiling_closure_args_t *args = (_pert_fcbias_fp32_tiling_closure_args_t *)_pert_fcbias_fp32_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_fcbias_fp32, Op: PerturbRademacher)
  uint8_t pert_fcbias_fp32_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcbias_fp32_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcbias_fp32_chunk = (9 >> pert_fcbias_fp32_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcbias_fp32_chunk_start = (uint32_t)MIN(pert_fcbias_fp32_chunk * pert_fcbias_fp32_core_id, (uint32_t)9);
  uint32_t pert_fcbias_fp32_chunk_stop = (uint32_t)MIN(pert_fcbias_fp32_chunk_start + pert_fcbias_fp32_chunk, (uint32_t)9);
  uint32_t pert_fcbias_fp32_local_size = pert_fcbias_fp32_chunk_stop - pert_fcbias_fp32_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_fcbias_fp32_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcbias_fp32_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref[pert_fcbias_fp32_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref[pert_fcbias_fp32_chunk_start], chunk_seed,
                            perturbation_sign, pert_fcbias_fp32_local_size, zo_eco_mom[11] + pert_fcbias_fp32_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref[pert_fcbias_fp32_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref[pert_fcbias_fp32_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_fcbias_fp32_local_size, pert_fcbias_fp32_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref[pert_fcbias_fp32_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref[pert_fcbias_fp32_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_fcbias_fp32_local_size, pert_fcbias_fp32_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref;
} _pert_fcbias_fp32_cluster_fork_args_t;

static void _pert_fcbias_fp32_cluster_fork(void *_pert_fcbias_fp32_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_cluster_fork_args_t *args = (_pert_fcbias_fp32_cluster_fork_args_t *)_pert_fcbias_fp32_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_tiling_closure_args_t DeeployNetwork__pert_fcbias_fp32_tiling_closure_args = (_pert_fcbias_fp32_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref};

  // _pert_fcbias_fp32_tiling_closure CLOSURE CALL
  _pert_fcbias_fp32_tiling_closure(&DeeployNetwork__pert_fcbias_fp32_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr;
} _pert_fcbias_fp32_closure_args_t;

static void _pert_fcbias_fp32_closure(void *_pert_fcbias_fp32_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_closure_args_t *args = (_pert_fcbias_fp32_closure_args_t *)_pert_fcbias_fp32_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor = args->DeeployNetwork_fcbias_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_input_23_ref =
      (void *)((char *)DeeployNetwork_input_23 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_fcbias_fp32_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr]);

  const static char pert_fcbias_fp32_L2_suffix[] = " cycles \n";

  const static char pert_fcbias_fp32_L2_prefix[] = "[pert_fcbias_fp32_L2][SB][9 ops][Tile ";

  static uint32_t pert_fcbias_fp32_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_fcbias_fp32_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_fcbias_fp32_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_fcbias_fp32_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_fcbias_fp32_L2_kernel_end_measurements[1];

  static uint32_t pert_fcbias_fp32_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr) + 1]; TILING_I++) {

    pert_fcbias_fp32_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_input_23_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_fcbias_fp32_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_fcbias_fp32_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_fcbias_fp32_cluster_fork_args_t DeeployNetwork__pert_fcbias_fp32_cluster_fork_args = (_pert_fcbias_fp32_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcbias_fp32_cluster_fork, &DeeployNetwork__pert_fcbias_fp32_cluster_fork_args);

    pert_fcbias_fp32_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_fcbias_fp32_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_fcbias_fp32_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_fcbias_fp32_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_fcbias_fp32_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr]; PROFILING_I++) {

    uint32_t pert_fcbias_fp32_L2_ingress_dma_wait_measurement =
        pert_fcbias_fp32_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_fcbias_fp32_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_fcbias_fp32_L2_kernel_measurement =
        pert_fcbias_fp32_L2_kernel_end_measurements[PROFILING_I] - pert_fcbias_fp32_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_fcbias_fp32_L2_egress_dma_wait_measurement =
        pert_fcbias_fp32_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_fcbias_fp32_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_fcbias_fp32_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_fcbias_fp32_L2_ingress_dma_wait_measurement,
           pert_fcbias_fp32_L2_suffix);

    printf("%s%u] %s%6u%s", pert_fcbias_fp32_L2_prefix, PROFILING_I, "Kernel     :", pert_fcbias_fp32_L2_kernel_measurement, pert_fcbias_fp32_L2_suffix);

    printf("%s%u] %s%6u%s", pert_fcbias_fp32_L2_prefix, PROFILING_I, "Post-Kernel:", pert_fcbias_fp32_L2_egress_dma_wait_measurement,
           pert_fcbias_fp32_L2_suffix);

    uint32_t total =
        pert_fcbias_fp32_L2_ingress_dma_wait_measurement + pert_fcbias_fp32_L2_kernel_measurement + pert_fcbias_fp32_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_fcbias_fp32_L2_ingress_dma_wait_measurement + pert_fcbias_fp32_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_fcbias_fp32_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_fcbias_fp32_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_fcbias_fp32_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr;
} _pert_fcbias_fp32_closure_L3_args_t;

static void _pert_fcbias_fp32_closure_L3(void *_pert_fcbias_fp32_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcbias_fp32_closure_L3_args_t *args = (_pert_fcbias_fp32_closure_L3_args_t *)_pert_fcbias_fp32_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor = args->DeeployNetwork_fcbias_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcbias_fp32_closure_args_t DeeployNetwork__pert_fcbias_fp32_closure_args = (_pert_fcbias_fp32_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_tensor = DeeployNetwork_fcbias_fp32_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr};

  // _pert_fcbias_fp32_closure CLOSURE CALL
  _pert_fcbias_fp32_closure(&DeeployNetwork__pert_fcbias_fp32_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks1convbias_rqsadd_tiling_closure(void *_rqsp_blocks1convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks1convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks1convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks1convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks1convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks1convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks1convbias_rqsadd_chunk = (16 >> rqsp_blocks1convbias_rqsadd_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks1convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks1convbias_rqsadd_chunk * rqsp_blocks1convbias_rqsadd_core_id, (uint32_t)16);
  uint32_t rqsp_blocks1convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks1convbias_rqsadd_chunk_start + rqsp_blocks1convbias_rqsadd_chunk, (uint32_t)16);
  uint32_t rqsp_blocks1convbias_rqsadd_local_size = rqsp_blocks1convbias_rqsadd_chunk_stop - rqsp_blocks1convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks1convbias_rqsadd_channel_start_offset = rqsp_blocks1convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 3 + rqsp_blocks1convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks1convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks1convbias_rqsadd_local_size,
        rqsp_blocks1convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[3] + rqsp_blocks1convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks1convbias_rqsadd_local_size,
        rqsp_blocks1convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks1convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks1convbias_rqsadd_cluster_fork(void *_rqsp_blocks1convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks1convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks1convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks1convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref};

  // _rqsp_blocks1convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks1convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks1convbias_rqsadd_closure_args_t;

static void _rqsp_blocks1convbias_rqsadd_closure(void *_rqsp_blocks1convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_closure_args_t *args = (_rqsp_blocks1convbias_rqsadd_closure_args_t *)_rqsp_blocks1convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_input_18_ref =
      (void *)((char *)DeeployNetwork_input_18 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);

  const static char rqsp_blocks1convbias_rqsadd_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks1convbias_rqsadd_L2_prefix[] = "[rqsp_blocks1convbias_rqsadd_L2][SB][16 ops][Tile ";

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks1convbias_rqsadd_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_input_18_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks1convbias_rqsadd_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks1convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks1convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks1convbias_rqsadd_cluster_fork_args);

    rqsp_blocks1convbias_rqsadd_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks1convbias_rqsadd_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_measurement = rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks1convbias_rqsadd_L2_kernel_measurement =
        rqsp_blocks1convbias_rqsadd_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks1convbias_rqsadd_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_measurement = rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks1convbias_rqsadd_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_measurement,
           rqsp_blocks1convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks1convbias_rqsadd_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks1convbias_rqsadd_L2_kernel_measurement,
           rqsp_blocks1convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks1convbias_rqsadd_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_measurement,
           rqsp_blocks1convbias_rqsadd_L2_suffix);

    uint32_t total = rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks1convbias_rqsadd_L2_kernel_measurement +
                     rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks1convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks1convbias_rqsadd_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks1convbias_rqsadd_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks1convbias_rqsadd_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks1convbias_rqsadd_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks1convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks1convbias_rqsadd_closure_L3(void *_rqsp_blocks1convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks1convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks1convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_args = (_rqsp_blocks1convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks1convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks1convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref;
} _rqsp_blocks2convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks2convbias_rqsadd_tiling_closure(void *_rqsp_blocks2convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks2convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks2convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks2convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks2convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks2convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks2convbias_rqsadd_chunk = (16 >> rqsp_blocks2convbias_rqsadd_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks2convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks2convbias_rqsadd_chunk * rqsp_blocks2convbias_rqsadd_core_id, (uint32_t)16);
  uint32_t rqsp_blocks2convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks2convbias_rqsadd_chunk_start + rqsp_blocks2convbias_rqsadd_chunk, (uint32_t)16);
  uint32_t rqsp_blocks2convbias_rqsadd_local_size = rqsp_blocks2convbias_rqsadd_chunk_stop - rqsp_blocks2convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks2convbias_rqsadd_channel_start_offset = rqsp_blocks2convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 5 + rqsp_blocks2convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks2convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref[rqsp_blocks2convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref[rqsp_blocks2convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks2convbias_rqsadd_local_size,
        rqsp_blocks2convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[5] + rqsp_blocks2convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref[rqsp_blocks2convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref[rqsp_blocks2convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks2convbias_rqsadd_local_size,
        rqsp_blocks2convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks2convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref;
} _rqsp_blocks2convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks2convbias_rqsadd_cluster_fork(void *_rqsp_blocks2convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks2convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks2convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks2convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks2convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks2convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref};

  // _rqsp_blocks2convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks2convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks2convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks2convbias_rqsadd_closure_args_t;

static void _rqsp_blocks2convbias_rqsadd_closure(void *_rqsp_blocks2convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convbias_rqsadd_closure_args_t *args = (_rqsp_blocks2convbias_rqsadd_closure_args_t *)_rqsp_blocks2convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_input_19_ref =
      (void *)((char *)DeeployNetwork_input_19 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr]);

  const static char rqsp_blocks2convbias_rqsadd_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks2convbias_rqsadd_L2_prefix[] = "[rqsp_blocks2convbias_rqsadd_L2][SB][16 ops][Tile ";

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks2convbias_rqsadd_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_input_19_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks2convbias_rqsadd_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks2convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks2convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks2convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks2convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks2convbias_rqsadd_cluster_fork_args);

    rqsp_blocks2convbias_rqsadd_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks2convbias_rqsadd_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_measurement = rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks2convbias_rqsadd_L2_kernel_measurement =
        rqsp_blocks2convbias_rqsadd_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks2convbias_rqsadd_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_measurement = rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks2convbias_rqsadd_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_measurement,
           rqsp_blocks2convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks2convbias_rqsadd_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks2convbias_rqsadd_L2_kernel_measurement,
           rqsp_blocks2convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks2convbias_rqsadd_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_measurement,
           rqsp_blocks2convbias_rqsadd_L2_suffix);

    uint32_t total = rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks2convbias_rqsadd_L2_kernel_measurement +
                     rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks2convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks2convbias_rqsadd_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks2convbias_rqsadd_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks2convbias_rqsadd_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks2convbias_rqsadd_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks2convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks2convbias_rqsadd_closure_L3(void *_rqsp_blocks2convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks2convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks2convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks2convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks2convbias_rqsadd_closure_args = (_rqsp_blocks2convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = DeeployNetwork_blocks2convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks2convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks2convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks2convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref;
} _pert_blocks1bnweight_tiling_closure_args_t;

static void _pert_blocks1bnweight_tiling_closure(void *_pert_blocks1bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_tiling_closure_args_t *args = (_pert_blocks1bnweight_tiling_closure_args_t *)_pert_blocks1bnweight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks1bnweight, Op: PerturbRademacher)
  uint8_t pert_blocks1bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnweight_chunk = (16 >> pert_blocks1bnweight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnweight_chunk_start = (uint32_t)MIN(pert_blocks1bnweight_chunk * pert_blocks1bnweight_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnweight_chunk_stop = (uint32_t)MIN(pert_blocks1bnweight_chunk_start + pert_blocks1bnweight_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnweight_local_size = pert_blocks1bnweight_chunk_stop - pert_blocks1bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 14 + pert_blocks1bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref[pert_blocks1bnweight_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref[pert_blocks1bnweight_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks1bnweight_local_size, zo_eco_mom[14] + pert_blocks1bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref[pert_blocks1bnweight_chunk_start],
        (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref[pert_blocks1bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_blocks1bnweight_local_size, pert_blocks1bnweight_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref[pert_blocks1bnweight_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref[pert_blocks1bnweight_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks1bnweight_local_size, pert_blocks1bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref;
} _pert_blocks1bnweight_cluster_fork_args_t;

static void _pert_blocks1bnweight_cluster_fork(void *_pert_blocks1bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_cluster_fork_args_t *args = (_pert_blocks1bnweight_cluster_fork_args_t *)_pert_blocks1bnweight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_tiling_closure_args_t DeeployNetwork__pert_blocks1bnweight_tiling_closure_args = (_pert_blocks1bnweight_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref};

  // _pert_blocks1bnweight_tiling_closure CLOSURE CALL
  _pert_blocks1bnweight_tiling_closure(&DeeployNetwork__pert_blocks1bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr;
} _pert_blocks1bnweight_closure_args_t;

static void _pert_blocks1bnweight_closure(void *_pert_blocks1bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_closure_args_t *args = (_pert_blocks1bnweight_closure_args_t *)_pert_blocks1bnweight_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor = args->DeeployNetwork_blocks1bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_input_4_ref =
      (void *)((char *)DeeployNetwork_input_4 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_blocks1bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr]);

  const static char pert_blocks1bnweight_L2_suffix[] = " cycles \n";

  const static char pert_blocks1bnweight_L2_prefix[] = "[pert_blocks1bnweight_L2][SB][16 ops][Tile ";

  static uint32_t pert_blocks1bnweight_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks1bnweight_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks1bnweight_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks1bnweight_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks1bnweight_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks1bnweight_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks1bnweight_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_input_4_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks1bnweight_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks1bnweight_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks1bnweight_cluster_fork_args_t DeeployNetwork__pert_blocks1bnweight_cluster_fork_args = (_pert_blocks1bnweight_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnweight_cluster_fork, &DeeployNetwork__pert_blocks1bnweight_cluster_fork_args);

    pert_blocks1bnweight_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks1bnweight_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_blocks1bnweight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks1bnweight_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks1bnweight_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks1bnweight_L2_ingress_dma_wait_measurement =
        pert_blocks1bnweight_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks1bnweight_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks1bnweight_L2_kernel_measurement =
        pert_blocks1bnweight_L2_kernel_end_measurements[PROFILING_I] - pert_blocks1bnweight_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks1bnweight_L2_egress_dma_wait_measurement =
        pert_blocks1bnweight_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks1bnweight_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks1bnweight_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks1bnweight_L2_ingress_dma_wait_measurement,
           pert_blocks1bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks1bnweight_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks1bnweight_L2_kernel_measurement,
           pert_blocks1bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks1bnweight_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks1bnweight_L2_egress_dma_wait_measurement,
           pert_blocks1bnweight_L2_suffix);

    uint32_t total =
        pert_blocks1bnweight_L2_ingress_dma_wait_measurement + pert_blocks1bnweight_L2_kernel_measurement + pert_blocks1bnweight_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks1bnweight_L2_ingress_dma_wait_measurement + pert_blocks1bnweight_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks1bnweight_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks1bnweight_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks1bnweight_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr;
} _pert_blocks1bnweight_closure_L3_args_t;

static void _pert_blocks1bnweight_closure_L3(void *_pert_blocks1bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnweight_closure_L3_args_t *args = (_pert_blocks1bnweight_closure_L3_args_t *)_pert_blocks1bnweight_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor = args->DeeployNetwork_blocks1bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnweight_closure_args_t DeeployNetwork__pert_blocks1bnweight_closure_args = (_pert_blocks1bnweight_closure_args_t){
      .DeeployNetwork_blocks1bnweight_pert_tensor = DeeployNetwork_blocks1bnweight_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr};

  // _pert_blocks1bnweight_closure CLOSURE CALL
  _pert_blocks1bnweight_closure(&DeeployNetwork__pert_blocks1bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref;
} _pert_blocks1bnbias_tiling_closure_args_t;

static void _pert_blocks1bnbias_tiling_closure(void *_pert_blocks1bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_tiling_closure_args_t *args = (_pert_blocks1bnbias_tiling_closure_args_t *)_pert_blocks1bnbias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks1bnbias, Op: PerturbRademacher)
  uint8_t pert_blocks1bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks1bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks1bnbias_chunk = (16 >> pert_blocks1bnbias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks1bnbias_chunk_start = (uint32_t)MIN(pert_blocks1bnbias_chunk * pert_blocks1bnbias_core_id, (uint32_t)16);
  uint32_t pert_blocks1bnbias_chunk_stop = (uint32_t)MIN(pert_blocks1bnbias_chunk_start + pert_blocks1bnbias_chunk, (uint32_t)16);
  uint32_t pert_blocks1bnbias_local_size = pert_blocks1bnbias_chunk_stop - pert_blocks1bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 15 + pert_blocks1bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks1bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref[pert_blocks1bnbias_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref[pert_blocks1bnbias_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks1bnbias_local_size, zo_eco_mom[15] + pert_blocks1bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref[pert_blocks1bnbias_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref[pert_blocks1bnbias_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_blocks1bnbias_local_size, pert_blocks1bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref[pert_blocks1bnbias_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref[pert_blocks1bnbias_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks1bnbias_local_size, pert_blocks1bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref;
} _pert_blocks1bnbias_cluster_fork_args_t;

static void _pert_blocks1bnbias_cluster_fork(void *_pert_blocks1bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_cluster_fork_args_t *args = (_pert_blocks1bnbias_cluster_fork_args_t *)_pert_blocks1bnbias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_tiling_closure_args_t DeeployNetwork__pert_blocks1bnbias_tiling_closure_args = (_pert_blocks1bnbias_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref};

  // _pert_blocks1bnbias_tiling_closure CLOSURE CALL
  _pert_blocks1bnbias_tiling_closure(&DeeployNetwork__pert_blocks1bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr;
} _pert_blocks1bnbias_closure_args_t;

static void _pert_blocks1bnbias_closure(void *_pert_blocks1bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_closure_args_t *args = (_pert_blocks1bnbias_closure_args_t *)_pert_blocks1bnbias_closure_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor = args->DeeployNetwork_blocks1bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_input_5_ref =
      (void *)((char *)DeeployNetwork_input_5 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_blocks1bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr]);

  const static char pert_blocks1bnbias_L2_suffix[] = " cycles \n";

  const static char pert_blocks1bnbias_L2_prefix[] = "[pert_blocks1bnbias_L2][SB][16 ops][Tile ";

  static uint32_t pert_blocks1bnbias_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks1bnbias_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks1bnbias_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks1bnbias_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks1bnbias_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks1bnbias_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks1bnbias_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_input_5_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks1bnbias_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks1bnbias_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks1bnbias_cluster_fork_args_t DeeployNetwork__pert_blocks1bnbias_cluster_fork_args = (_pert_blocks1bnbias_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks1bnbias_cluster_fork, &DeeployNetwork__pert_blocks1bnbias_cluster_fork_args);

    pert_blocks1bnbias_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks1bnbias_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_blocks1bnbias_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks1bnbias_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks1bnbias_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks1bnbias_L2_ingress_dma_wait_measurement =
        pert_blocks1bnbias_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks1bnbias_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks1bnbias_L2_kernel_measurement =
        pert_blocks1bnbias_L2_kernel_end_measurements[PROFILING_I] - pert_blocks1bnbias_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks1bnbias_L2_egress_dma_wait_measurement =
        pert_blocks1bnbias_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks1bnbias_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks1bnbias_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks1bnbias_L2_ingress_dma_wait_measurement,
           pert_blocks1bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks1bnbias_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks1bnbias_L2_kernel_measurement, pert_blocks1bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks1bnbias_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks1bnbias_L2_egress_dma_wait_measurement,
           pert_blocks1bnbias_L2_suffix);

    uint32_t total =
        pert_blocks1bnbias_L2_ingress_dma_wait_measurement + pert_blocks1bnbias_L2_kernel_measurement + pert_blocks1bnbias_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks1bnbias_L2_ingress_dma_wait_measurement + pert_blocks1bnbias_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks1bnbias_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks1bnbias_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks1bnbias_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr;
} _pert_blocks1bnbias_closure_L3_args_t;

static void _pert_blocks1bnbias_closure_L3(void *_pert_blocks1bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks1bnbias_closure_L3_args_t *args = (_pert_blocks1bnbias_closure_L3_args_t *)_pert_blocks1bnbias_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor = args->DeeployNetwork_blocks1bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks1bnbias_closure_args_t DeeployNetwork__pert_blocks1bnbias_closure_args = (_pert_blocks1bnbias_closure_args_t){
      .DeeployNetwork_blocks1bnbias_pert_tensor = DeeployNetwork_blocks1bnbias_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr};

  // _pert_blocks1bnbias_closure CLOSURE CALL
  _pert_blocks1bnbias_closure(&DeeployNetwork__pert_blocks1bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref;
} _pert_blocks2bnweight_tiling_closure_args_t;

static void _pert_blocks2bnweight_tiling_closure(void *_pert_blocks2bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_tiling_closure_args_t *args = (_pert_blocks2bnweight_tiling_closure_args_t *)_pert_blocks2bnweight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks2bnweight, Op: PerturbRademacher)
  uint8_t pert_blocks2bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnweight_chunk = (16 >> pert_blocks2bnweight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnweight_chunk_start = (uint32_t)MIN(pert_blocks2bnweight_chunk * pert_blocks2bnweight_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnweight_chunk_stop = (uint32_t)MIN(pert_blocks2bnweight_chunk_start + pert_blocks2bnweight_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnweight_local_size = pert_blocks2bnweight_chunk_stop - pert_blocks2bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 16 + pert_blocks2bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref[pert_blocks2bnweight_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref[pert_blocks2bnweight_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks2bnweight_local_size, zo_eco_mom[16] + pert_blocks2bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref[pert_blocks2bnweight_chunk_start],
        (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref[pert_blocks2bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_blocks2bnweight_local_size, pert_blocks2bnweight_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref[pert_blocks2bnweight_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref[pert_blocks2bnweight_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks2bnweight_local_size, pert_blocks2bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref;
} _pert_blocks2bnweight_cluster_fork_args_t;

static void _pert_blocks2bnweight_cluster_fork(void *_pert_blocks2bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_cluster_fork_args_t *args = (_pert_blocks2bnweight_cluster_fork_args_t *)_pert_blocks2bnweight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_tiling_closure_args_t DeeployNetwork__pert_blocks2bnweight_tiling_closure_args = (_pert_blocks2bnweight_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref};

  // _pert_blocks2bnweight_tiling_closure CLOSURE CALL
  _pert_blocks2bnweight_tiling_closure(&DeeployNetwork__pert_blocks2bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr;
} _pert_blocks2bnweight_closure_args_t;

static void _pert_blocks2bnweight_closure(void *_pert_blocks2bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_closure_args_t *args = (_pert_blocks2bnweight_closure_args_t *)_pert_blocks2bnweight_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor = args->DeeployNetwork_blocks2bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_input_6_ref =
      (void *)((char *)DeeployNetwork_input_6 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_blocks2bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr]);

  const static char pert_blocks2bnweight_L2_suffix[] = " cycles \n";

  const static char pert_blocks2bnweight_L2_prefix[] = "[pert_blocks2bnweight_L2][SB][16 ops][Tile ";

  static uint32_t pert_blocks2bnweight_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks2bnweight_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks2bnweight_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks2bnweight_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks2bnweight_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks2bnweight_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks2bnweight_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_input_6_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks2bnweight_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks2bnweight_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks2bnweight_cluster_fork_args_t DeeployNetwork__pert_blocks2bnweight_cluster_fork_args = (_pert_blocks2bnweight_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnweight_cluster_fork, &DeeployNetwork__pert_blocks2bnweight_cluster_fork_args);

    pert_blocks2bnweight_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks2bnweight_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_blocks2bnweight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks2bnweight_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks2bnweight_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks2bnweight_L2_ingress_dma_wait_measurement =
        pert_blocks2bnweight_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks2bnweight_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks2bnweight_L2_kernel_measurement =
        pert_blocks2bnweight_L2_kernel_end_measurements[PROFILING_I] - pert_blocks2bnweight_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks2bnweight_L2_egress_dma_wait_measurement =
        pert_blocks2bnweight_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks2bnweight_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks2bnweight_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks2bnweight_L2_ingress_dma_wait_measurement,
           pert_blocks2bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks2bnweight_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks2bnweight_L2_kernel_measurement,
           pert_blocks2bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks2bnweight_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks2bnweight_L2_egress_dma_wait_measurement,
           pert_blocks2bnweight_L2_suffix);

    uint32_t total =
        pert_blocks2bnweight_L2_ingress_dma_wait_measurement + pert_blocks2bnweight_L2_kernel_measurement + pert_blocks2bnweight_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks2bnweight_L2_ingress_dma_wait_measurement + pert_blocks2bnweight_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks2bnweight_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks2bnweight_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks2bnweight_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr;
} _pert_blocks2bnweight_closure_L3_args_t;

static void _pert_blocks2bnweight_closure_L3(void *_pert_blocks2bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnweight_closure_L3_args_t *args = (_pert_blocks2bnweight_closure_L3_args_t *)_pert_blocks2bnweight_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor = args->DeeployNetwork_blocks2bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnweight_closure_args_t DeeployNetwork__pert_blocks2bnweight_closure_args = (_pert_blocks2bnweight_closure_args_t){
      .DeeployNetwork_blocks2bnweight_pert_tensor = DeeployNetwork_blocks2bnweight_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr};

  // _pert_blocks2bnweight_closure CLOSURE CALL
  _pert_blocks2bnweight_closure(&DeeployNetwork__pert_blocks2bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref;
} _pert_blocks2bnbias_tiling_closure_args_t;

static void _pert_blocks2bnbias_tiling_closure(void *_pert_blocks2bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_tiling_closure_args_t *args = (_pert_blocks2bnbias_tiling_closure_args_t *)_pert_blocks2bnbias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks2bnbias, Op: PerturbRademacher)
  uint8_t pert_blocks2bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks2bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks2bnbias_chunk = (16 >> pert_blocks2bnbias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks2bnbias_chunk_start = (uint32_t)MIN(pert_blocks2bnbias_chunk * pert_blocks2bnbias_core_id, (uint32_t)16);
  uint32_t pert_blocks2bnbias_chunk_stop = (uint32_t)MIN(pert_blocks2bnbias_chunk_start + pert_blocks2bnbias_chunk, (uint32_t)16);
  uint32_t pert_blocks2bnbias_local_size = pert_blocks2bnbias_chunk_stop - pert_blocks2bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 17 + pert_blocks2bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks2bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref[pert_blocks2bnbias_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref[pert_blocks2bnbias_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks2bnbias_local_size, zo_eco_mom[17] + pert_blocks2bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref[pert_blocks2bnbias_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref[pert_blocks2bnbias_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_blocks2bnbias_local_size, pert_blocks2bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref[pert_blocks2bnbias_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref[pert_blocks2bnbias_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks2bnbias_local_size, pert_blocks2bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref;
} _pert_blocks2bnbias_cluster_fork_args_t;

static void _pert_blocks2bnbias_cluster_fork(void *_pert_blocks2bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_cluster_fork_args_t *args = (_pert_blocks2bnbias_cluster_fork_args_t *)_pert_blocks2bnbias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_tiling_closure_args_t DeeployNetwork__pert_blocks2bnbias_tiling_closure_args = (_pert_blocks2bnbias_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref};

  // _pert_blocks2bnbias_tiling_closure CLOSURE CALL
  _pert_blocks2bnbias_tiling_closure(&DeeployNetwork__pert_blocks2bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr;
} _pert_blocks2bnbias_closure_args_t;

static void _pert_blocks2bnbias_closure(void *_pert_blocks2bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_closure_args_t *args = (_pert_blocks2bnbias_closure_args_t *)_pert_blocks2bnbias_closure_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor = args->DeeployNetwork_blocks2bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_input_7_ref =
      (void *)((char *)DeeployNetwork_input_7 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_blocks2bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr]);

  const static char pert_blocks2bnbias_L2_suffix[] = " cycles \n";

  const static char pert_blocks2bnbias_L2_prefix[] = "[pert_blocks2bnbias_L2][SB][16 ops][Tile ";

  static uint32_t pert_blocks2bnbias_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks2bnbias_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks2bnbias_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks2bnbias_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks2bnbias_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks2bnbias_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks2bnbias_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_input_7_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks2bnbias_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks2bnbias_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks2bnbias_cluster_fork_args_t DeeployNetwork__pert_blocks2bnbias_cluster_fork_args = (_pert_blocks2bnbias_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks2bnbias_cluster_fork, &DeeployNetwork__pert_blocks2bnbias_cluster_fork_args);

    pert_blocks2bnbias_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks2bnbias_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_blocks2bnbias_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks2bnbias_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks2bnbias_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks2bnbias_L2_ingress_dma_wait_measurement =
        pert_blocks2bnbias_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks2bnbias_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks2bnbias_L2_kernel_measurement =
        pert_blocks2bnbias_L2_kernel_end_measurements[PROFILING_I] - pert_blocks2bnbias_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks2bnbias_L2_egress_dma_wait_measurement =
        pert_blocks2bnbias_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks2bnbias_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks2bnbias_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks2bnbias_L2_ingress_dma_wait_measurement,
           pert_blocks2bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks2bnbias_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks2bnbias_L2_kernel_measurement, pert_blocks2bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks2bnbias_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks2bnbias_L2_egress_dma_wait_measurement,
           pert_blocks2bnbias_L2_suffix);

    uint32_t total =
        pert_blocks2bnbias_L2_ingress_dma_wait_measurement + pert_blocks2bnbias_L2_kernel_measurement + pert_blocks2bnbias_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks2bnbias_L2_ingress_dma_wait_measurement + pert_blocks2bnbias_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks2bnbias_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks2bnbias_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks2bnbias_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr;
} _pert_blocks2bnbias_closure_L3_args_t;

static void _pert_blocks2bnbias_closure_L3(void *_pert_blocks2bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks2bnbias_closure_L3_args_t *args = (_pert_blocks2bnbias_closure_L3_args_t *)_pert_blocks2bnbias_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor = args->DeeployNetwork_blocks2bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks2bnbias_closure_args_t DeeployNetwork__pert_blocks2bnbias_closure_args = (_pert_blocks2bnbias_closure_args_t){
      .DeeployNetwork_blocks2bnbias_pert_tensor = DeeployNetwork_blocks2bnbias_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr};

  // _pert_blocks2bnbias_closure CLOSURE CALL
  _pert_blocks2bnbias_closure(&DeeployNetwork__pert_blocks2bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;
} _rqsp_blocks0convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks0convweight_int8_tiling_closure(void *_rqsp_blocks0convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks0convweight_int8_tiling_closure_args_t *)_rqsp_blocks0convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks0convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks0convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks0convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks0convweight_int8_chunk = (32 >> rqsp_blocks0convweight_int8_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks0convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks0convweight_int8_chunk * rqsp_blocks0convweight_int8_core_id, (uint32_t)32);
  uint32_t rqsp_blocks0convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks0convweight_int8_chunk_start + rqsp_blocks0convweight_int8_chunk, (uint32_t)32);
  uint32_t rqsp_blocks0convweight_int8_local_size = rqsp_blocks0convweight_int8_chunk_stop - rqsp_blocks0convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks0convweight_int8_channel_start_offset = rqsp_blocks0convweight_int8_chunk_start % 4;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + rqsp_blocks0convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks0convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref[rqsp_blocks0convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref[rqsp_blocks0convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref, 15, 4, chunk_seed, rqsp_blocks0convweight_int8_local_size,
        rqsp_blocks0convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[0] + rqsp_blocks0convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref[rqsp_blocks0convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref[rqsp_blocks0convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref, 15, 4, chunk_seed, rqsp_blocks0convweight_int8_local_size,
        rqsp_blocks0convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks0convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;
} _rqsp_blocks0convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks0convweight_int8_cluster_fork(void *_rqsp_blocks0convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks0convweight_int8_cluster_fork_args_t *)_rqsp_blocks0convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks0convweight_int8_tiling_closure_args =
      (_rqsp_blocks0convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref};

  // _rqsp_blocks0convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks0convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks0convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
} _rqsp_blocks0convweight_int8_closure_args_t;

static void _rqsp_blocks0convweight_int8_closure(void *_rqsp_blocks0convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_closure_args_t *args = (_rqsp_blocks0convweight_int8_closure_args_t *)_rqsp_blocks0convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_input_12_ref =
      (void *)((char *)DeeployNetwork_input_12 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);

  const static char rqsp_blocks0convweight_int8_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks0convweight_int8_L2_prefix[] = "[rqsp_blocks0convweight_int8_L2][SB][32 ops][Tile ";

  static uint32_t rqsp_blocks0convweight_int8_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks0convweight_int8_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks0convweight_int8_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks0convweight_int8_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks0convweight_int8_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks0convweight_int8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks0convweight_int8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_input_12_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks0convweight_int8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks0convweight_int8_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks0convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks0convweight_int8_cluster_fork_args =
        (_rqsp_blocks0convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks0convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks0convweight_int8_cluster_fork_args);

    rqsp_blocks0convweight_int8_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks0convweight_int8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks0convweight_int8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks0convweight_int8_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks0convweight_int8_L2_ingress_dma_wait_measurement = rqsp_blocks0convweight_int8_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks0convweight_int8_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks0convweight_int8_L2_kernel_measurement =
        rqsp_blocks0convweight_int8_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks0convweight_int8_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks0convweight_int8_L2_egress_dma_wait_measurement = rqsp_blocks0convweight_int8_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks0convweight_int8_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks0convweight_int8_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks0convweight_int8_L2_ingress_dma_wait_measurement,
           rqsp_blocks0convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks0convweight_int8_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks0convweight_int8_L2_kernel_measurement,
           rqsp_blocks0convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks0convweight_int8_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks0convweight_int8_L2_egress_dma_wait_measurement,
           rqsp_blocks0convweight_int8_L2_suffix);

    uint32_t total = rqsp_blocks0convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks0convweight_int8_L2_kernel_measurement +
                     rqsp_blocks0convweight_int8_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks0convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks0convweight_int8_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks0convweight_int8_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks0convweight_int8_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks0convweight_int8_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
} _rqsp_blocks0convweight_int8_closure_L3_args_t;

static void _rqsp_blocks0convweight_int8_closure_L3(void *_rqsp_blocks0convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_closure_L3_args_t *args = (_rqsp_blocks0convweight_int8_closure_L3_args_t *)_rqsp_blocks0convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks0convweight_int8_closure_args = (_rqsp_blocks0convweight_int8_closure_args_t){
      .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr};

  // _rqsp_blocks0convweight_int8_closure CLOSURE CALL
  _rqsp_blocks0convweight_int8_closure(&DeeployNetwork__rqsp_blocks0convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [8, 1, 1, 4] -> [8, 1, 4, 1] (Name: _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 8;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_blocks0convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441824,
          .size = 32,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_blocks0convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1310752,
          .size = 32,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks3convbias_rqsadd_tiling_closure(void *_rqsp_blocks3convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks3convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks3convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks3convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks3convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks3convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks3convbias_rqsadd_chunk = (32 >> rqsp_blocks3convbias_rqsadd_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks3convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks3convbias_rqsadd_chunk * rqsp_blocks3convbias_rqsadd_core_id, (uint32_t)32);
  uint32_t rqsp_blocks3convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks3convbias_rqsadd_chunk_start + rqsp_blocks3convbias_rqsadd_chunk, (uint32_t)32);
  uint32_t rqsp_blocks3convbias_rqsadd_local_size = rqsp_blocks3convbias_rqsadd_chunk_stop - rqsp_blocks3convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks3convbias_rqsadd_channel_start_offset = rqsp_blocks3convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 7 + rqsp_blocks3convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks3convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks3convbias_rqsadd_local_size,
        rqsp_blocks3convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[7] + rqsp_blocks3convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks3convbias_rqsadd_local_size,
        rqsp_blocks3convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks3convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks3convbias_rqsadd_cluster_fork(void *_rqsp_blocks3convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks3convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks3convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks3convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref};

  // _rqsp_blocks3convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks3convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks3convbias_rqsadd_closure_args_t;

static void _rqsp_blocks3convbias_rqsadd_closure(void *_rqsp_blocks3convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_closure_args_t *args = (_rqsp_blocks3convbias_rqsadd_closure_args_t *)_rqsp_blocks3convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_input_20_ref =
      (void *)((char *)DeeployNetwork_input_20 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);

  const static char rqsp_blocks3convbias_rqsadd_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks3convbias_rqsadd_L2_prefix[] = "[rqsp_blocks3convbias_rqsadd_L2][SB][32 ops][Tile ";

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks3convbias_rqsadd_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_input_20_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks3convbias_rqsadd_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks3convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks3convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks3convbias_rqsadd_cluster_fork_args);

    rqsp_blocks3convbias_rqsadd_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks3convbias_rqsadd_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_measurement = rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks3convbias_rqsadd_L2_kernel_measurement =
        rqsp_blocks3convbias_rqsadd_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks3convbias_rqsadd_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_measurement = rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks3convbias_rqsadd_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_measurement,
           rqsp_blocks3convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks3convbias_rqsadd_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks3convbias_rqsadd_L2_kernel_measurement,
           rqsp_blocks3convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks3convbias_rqsadd_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_measurement,
           rqsp_blocks3convbias_rqsadd_L2_suffix);

    uint32_t total = rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks3convbias_rqsadd_L2_kernel_measurement +
                     rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks3convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks3convbias_rqsadd_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks3convbias_rqsadd_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks3convbias_rqsadd_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks3convbias_rqsadd_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks3convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks3convbias_rqsadd_closure_L3(void *_rqsp_blocks3convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks3convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks3convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_args = (_rqsp_blocks3convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks3convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks3convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref;
} _rqsp_blocks4convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks4convbias_rqsadd_tiling_closure(void *_rqsp_blocks4convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks4convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks4convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks4convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks4convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks4convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks4convbias_rqsadd_chunk = (32 >> rqsp_blocks4convbias_rqsadd_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks4convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks4convbias_rqsadd_chunk * rqsp_blocks4convbias_rqsadd_core_id, (uint32_t)32);
  uint32_t rqsp_blocks4convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks4convbias_rqsadd_chunk_start + rqsp_blocks4convbias_rqsadd_chunk, (uint32_t)32);
  uint32_t rqsp_blocks4convbias_rqsadd_local_size = rqsp_blocks4convbias_rqsadd_chunk_stop - rqsp_blocks4convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks4convbias_rqsadd_channel_start_offset = rqsp_blocks4convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 9 + rqsp_blocks4convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks4convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref[rqsp_blocks4convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref[rqsp_blocks4convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks4convbias_rqsadd_local_size,
        rqsp_blocks4convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[9] + rqsp_blocks4convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref[rqsp_blocks4convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref[rqsp_blocks4convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks4convbias_rqsadd_local_size,
        rqsp_blocks4convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks4convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref;
} _rqsp_blocks4convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks4convbias_rqsadd_cluster_fork(void *_rqsp_blocks4convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks4convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks4convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks4convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks4convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks4convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref};

  // _rqsp_blocks4convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks4convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks4convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks4convbias_rqsadd_closure_args_t;

static void _rqsp_blocks4convbias_rqsadd_closure(void *_rqsp_blocks4convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convbias_rqsadd_closure_args_t *args = (_rqsp_blocks4convbias_rqsadd_closure_args_t *)_rqsp_blocks4convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_input_21_ref =
      (void *)((char *)DeeployNetwork_input_21 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr]);

  const static char rqsp_blocks4convbias_rqsadd_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks4convbias_rqsadd_L2_prefix[] = "[rqsp_blocks4convbias_rqsadd_L2][SB][32 ops][Tile ";

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks4convbias_rqsadd_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_input_21_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks4convbias_rqsadd_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks4convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks4convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks4convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks4convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks4convbias_rqsadd_cluster_fork_args);

    rqsp_blocks4convbias_rqsadd_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks4convbias_rqsadd_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_measurement = rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks4convbias_rqsadd_L2_kernel_measurement =
        rqsp_blocks4convbias_rqsadd_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks4convbias_rqsadd_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_measurement = rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks4convbias_rqsadd_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_measurement,
           rqsp_blocks4convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks4convbias_rqsadd_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks4convbias_rqsadd_L2_kernel_measurement,
           rqsp_blocks4convbias_rqsadd_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks4convbias_rqsadd_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_measurement,
           rqsp_blocks4convbias_rqsadd_L2_suffix);

    uint32_t total = rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks4convbias_rqsadd_L2_kernel_measurement +
                     rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks4convbias_rqsadd_L2_ingress_dma_wait_measurement + rqsp_blocks4convbias_rqsadd_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks4convbias_rqsadd_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks4convbias_rqsadd_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks4convbias_rqsadd_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks4convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks4convbias_rqsadd_closure_L3(void *_rqsp_blocks4convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks4convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks4convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks4convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks4convbias_rqsadd_closure_args = (_rqsp_blocks4convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = DeeployNetwork_blocks4convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks4convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks4convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks4convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref;
} _pert_blocks3bnweight_tiling_closure_args_t;

static void _pert_blocks3bnweight_tiling_closure(void *_pert_blocks3bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_tiling_closure_args_t *args = (_pert_blocks3bnweight_tiling_closure_args_t *)_pert_blocks3bnweight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks3bnweight, Op: PerturbRademacher)
  uint8_t pert_blocks3bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnweight_chunk = (32 >> pert_blocks3bnweight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnweight_chunk_start = (uint32_t)MIN(pert_blocks3bnweight_chunk * pert_blocks3bnweight_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnweight_chunk_stop = (uint32_t)MIN(pert_blocks3bnweight_chunk_start + pert_blocks3bnweight_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnweight_local_size = pert_blocks3bnweight_chunk_stop - pert_blocks3bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 18 + pert_blocks3bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref[pert_blocks3bnweight_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref[pert_blocks3bnweight_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks3bnweight_local_size, zo_eco_mom[18] + pert_blocks3bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref[pert_blocks3bnweight_chunk_start],
        (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref[pert_blocks3bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_blocks3bnweight_local_size, pert_blocks3bnweight_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref[pert_blocks3bnweight_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref[pert_blocks3bnweight_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks3bnweight_local_size, pert_blocks3bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref;
} _pert_blocks3bnweight_cluster_fork_args_t;

static void _pert_blocks3bnweight_cluster_fork(void *_pert_blocks3bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_cluster_fork_args_t *args = (_pert_blocks3bnweight_cluster_fork_args_t *)_pert_blocks3bnweight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_tiling_closure_args_t DeeployNetwork__pert_blocks3bnweight_tiling_closure_args = (_pert_blocks3bnweight_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref};

  // _pert_blocks3bnweight_tiling_closure CLOSURE CALL
  _pert_blocks3bnweight_tiling_closure(&DeeployNetwork__pert_blocks3bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr;
} _pert_blocks3bnweight_closure_args_t;

static void _pert_blocks3bnweight_closure(void *_pert_blocks3bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_closure_args_t *args = (_pert_blocks3bnweight_closure_args_t *)_pert_blocks3bnweight_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor = args->DeeployNetwork_blocks3bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_input_8_ref =
      (void *)((char *)DeeployNetwork_input_8 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_blocks3bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr]);

  const static char pert_blocks3bnweight_L2_suffix[] = " cycles \n";

  const static char pert_blocks3bnweight_L2_prefix[] = "[pert_blocks3bnweight_L2][SB][32 ops][Tile ";

  static uint32_t pert_blocks3bnweight_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks3bnweight_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks3bnweight_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks3bnweight_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks3bnweight_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks3bnweight_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks3bnweight_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_input_8_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks3bnweight_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks3bnweight_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks3bnweight_cluster_fork_args_t DeeployNetwork__pert_blocks3bnweight_cluster_fork_args = (_pert_blocks3bnweight_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnweight_cluster_fork, &DeeployNetwork__pert_blocks3bnweight_cluster_fork_args);

    pert_blocks3bnweight_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks3bnweight_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_blocks3bnweight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks3bnweight_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks3bnweight_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks3bnweight_L2_ingress_dma_wait_measurement =
        pert_blocks3bnweight_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks3bnweight_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks3bnweight_L2_kernel_measurement =
        pert_blocks3bnweight_L2_kernel_end_measurements[PROFILING_I] - pert_blocks3bnweight_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks3bnweight_L2_egress_dma_wait_measurement =
        pert_blocks3bnweight_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks3bnweight_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks3bnweight_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks3bnweight_L2_ingress_dma_wait_measurement,
           pert_blocks3bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks3bnweight_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks3bnweight_L2_kernel_measurement,
           pert_blocks3bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks3bnweight_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks3bnweight_L2_egress_dma_wait_measurement,
           pert_blocks3bnweight_L2_suffix);

    uint32_t total =
        pert_blocks3bnweight_L2_ingress_dma_wait_measurement + pert_blocks3bnweight_L2_kernel_measurement + pert_blocks3bnweight_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks3bnweight_L2_ingress_dma_wait_measurement + pert_blocks3bnweight_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks3bnweight_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks3bnweight_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks3bnweight_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr;
} _pert_blocks3bnweight_closure_L3_args_t;

static void _pert_blocks3bnweight_closure_L3(void *_pert_blocks3bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnweight_closure_L3_args_t *args = (_pert_blocks3bnweight_closure_L3_args_t *)_pert_blocks3bnweight_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor = args->DeeployNetwork_blocks3bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnweight_closure_args_t DeeployNetwork__pert_blocks3bnweight_closure_args = (_pert_blocks3bnweight_closure_args_t){
      .DeeployNetwork_blocks3bnweight_pert_tensor = DeeployNetwork_blocks3bnweight_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr};

  // _pert_blocks3bnweight_closure CLOSURE CALL
  _pert_blocks3bnweight_closure(&DeeployNetwork__pert_blocks3bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref;
} _pert_blocks3bnbias_tiling_closure_args_t;

static void _pert_blocks3bnbias_tiling_closure(void *_pert_blocks3bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_tiling_closure_args_t *args = (_pert_blocks3bnbias_tiling_closure_args_t *)_pert_blocks3bnbias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks3bnbias, Op: PerturbRademacher)
  uint8_t pert_blocks3bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks3bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks3bnbias_chunk = (32 >> pert_blocks3bnbias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks3bnbias_chunk_start = (uint32_t)MIN(pert_blocks3bnbias_chunk * pert_blocks3bnbias_core_id, (uint32_t)32);
  uint32_t pert_blocks3bnbias_chunk_stop = (uint32_t)MIN(pert_blocks3bnbias_chunk_start + pert_blocks3bnbias_chunk, (uint32_t)32);
  uint32_t pert_blocks3bnbias_local_size = pert_blocks3bnbias_chunk_stop - pert_blocks3bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 19 + pert_blocks3bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks3bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref[pert_blocks3bnbias_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref[pert_blocks3bnbias_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks3bnbias_local_size, zo_eco_mom[19] + pert_blocks3bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref[pert_blocks3bnbias_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref[pert_blocks3bnbias_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_blocks3bnbias_local_size, pert_blocks3bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref[pert_blocks3bnbias_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref[pert_blocks3bnbias_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks3bnbias_local_size, pert_blocks3bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref;
} _pert_blocks3bnbias_cluster_fork_args_t;

static void _pert_blocks3bnbias_cluster_fork(void *_pert_blocks3bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_cluster_fork_args_t *args = (_pert_blocks3bnbias_cluster_fork_args_t *)_pert_blocks3bnbias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_tiling_closure_args_t DeeployNetwork__pert_blocks3bnbias_tiling_closure_args = (_pert_blocks3bnbias_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref};

  // _pert_blocks3bnbias_tiling_closure CLOSURE CALL
  _pert_blocks3bnbias_tiling_closure(&DeeployNetwork__pert_blocks3bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr;
} _pert_blocks3bnbias_closure_args_t;

static void _pert_blocks3bnbias_closure(void *_pert_blocks3bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_closure_args_t *args = (_pert_blocks3bnbias_closure_args_t *)_pert_blocks3bnbias_closure_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor = args->DeeployNetwork_blocks3bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_input_9_ref =
      (void *)((char *)DeeployNetwork_input_9 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_blocks3bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr]);

  const static char pert_blocks3bnbias_L2_suffix[] = " cycles \n";

  const static char pert_blocks3bnbias_L2_prefix[] = "[pert_blocks3bnbias_L2][SB][32 ops][Tile ";

  static uint32_t pert_blocks3bnbias_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks3bnbias_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks3bnbias_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks3bnbias_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks3bnbias_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks3bnbias_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks3bnbias_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_input_9_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks3bnbias_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks3bnbias_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks3bnbias_cluster_fork_args_t DeeployNetwork__pert_blocks3bnbias_cluster_fork_args = (_pert_blocks3bnbias_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks3bnbias_cluster_fork, &DeeployNetwork__pert_blocks3bnbias_cluster_fork_args);

    pert_blocks3bnbias_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks3bnbias_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_blocks3bnbias_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks3bnbias_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks3bnbias_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks3bnbias_L2_ingress_dma_wait_measurement =
        pert_blocks3bnbias_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks3bnbias_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks3bnbias_L2_kernel_measurement =
        pert_blocks3bnbias_L2_kernel_end_measurements[PROFILING_I] - pert_blocks3bnbias_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks3bnbias_L2_egress_dma_wait_measurement =
        pert_blocks3bnbias_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks3bnbias_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks3bnbias_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks3bnbias_L2_ingress_dma_wait_measurement,
           pert_blocks3bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks3bnbias_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks3bnbias_L2_kernel_measurement, pert_blocks3bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks3bnbias_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks3bnbias_L2_egress_dma_wait_measurement,
           pert_blocks3bnbias_L2_suffix);

    uint32_t total =
        pert_blocks3bnbias_L2_ingress_dma_wait_measurement + pert_blocks3bnbias_L2_kernel_measurement + pert_blocks3bnbias_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks3bnbias_L2_ingress_dma_wait_measurement + pert_blocks3bnbias_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks3bnbias_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks3bnbias_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks3bnbias_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr;
} _pert_blocks3bnbias_closure_L3_args_t;

static void _pert_blocks3bnbias_closure_L3(void *_pert_blocks3bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks3bnbias_closure_L3_args_t *args = (_pert_blocks3bnbias_closure_L3_args_t *)_pert_blocks3bnbias_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor = args->DeeployNetwork_blocks3bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks3bnbias_closure_args_t DeeployNetwork__pert_blocks3bnbias_closure_args = (_pert_blocks3bnbias_closure_args_t){
      .DeeployNetwork_blocks3bnbias_pert_tensor = DeeployNetwork_blocks3bnbias_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr};

  // _pert_blocks3bnbias_closure CLOSURE CALL
  _pert_blocks3bnbias_closure(&DeeployNetwork__pert_blocks3bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref;
} _pert_blocks4bnweight_tiling_closure_args_t;

static void _pert_blocks4bnweight_tiling_closure(void *_pert_blocks4bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_tiling_closure_args_t *args = (_pert_blocks4bnweight_tiling_closure_args_t *)_pert_blocks4bnweight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks4bnweight, Op: PerturbRademacher)
  uint8_t pert_blocks4bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnweight_chunk = (32 >> pert_blocks4bnweight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnweight_chunk_start = (uint32_t)MIN(pert_blocks4bnweight_chunk * pert_blocks4bnweight_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnweight_chunk_stop = (uint32_t)MIN(pert_blocks4bnweight_chunk_start + pert_blocks4bnweight_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnweight_local_size = pert_blocks4bnweight_chunk_stop - pert_blocks4bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 20 + pert_blocks4bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref[pert_blocks4bnweight_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref[pert_blocks4bnweight_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks4bnweight_local_size, zo_eco_mom[20] + pert_blocks4bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref[pert_blocks4bnweight_chunk_start],
        (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref[pert_blocks4bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_blocks4bnweight_local_size, pert_blocks4bnweight_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref[pert_blocks4bnweight_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref[pert_blocks4bnweight_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks4bnweight_local_size, pert_blocks4bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref;
} _pert_blocks4bnweight_cluster_fork_args_t;

static void _pert_blocks4bnweight_cluster_fork(void *_pert_blocks4bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_cluster_fork_args_t *args = (_pert_blocks4bnweight_cluster_fork_args_t *)_pert_blocks4bnweight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_tiling_closure_args_t DeeployNetwork__pert_blocks4bnweight_tiling_closure_args = (_pert_blocks4bnweight_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref};

  // _pert_blocks4bnweight_tiling_closure CLOSURE CALL
  _pert_blocks4bnweight_tiling_closure(&DeeployNetwork__pert_blocks4bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr;
} _pert_blocks4bnweight_closure_args_t;

static void _pert_blocks4bnweight_closure(void *_pert_blocks4bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_closure_args_t *args = (_pert_blocks4bnweight_closure_args_t *)_pert_blocks4bnweight_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor = args->DeeployNetwork_blocks4bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_input_10_ref =
      (void *)((char *)DeeployNetwork_input_10 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_blocks4bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnweight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr]);

  const static char pert_blocks4bnweight_L2_suffix[] = " cycles \n";

  const static char pert_blocks4bnweight_L2_prefix[] = "[pert_blocks4bnweight_L2][SB][32 ops][Tile ";

  static uint32_t pert_blocks4bnweight_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks4bnweight_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks4bnweight_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks4bnweight_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks4bnweight_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks4bnweight_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks4bnweight_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_input_10_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks4bnweight_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks4bnweight_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks4bnweight_cluster_fork_args_t DeeployNetwork__pert_blocks4bnweight_cluster_fork_args = (_pert_blocks4bnweight_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnweight_cluster_fork, &DeeployNetwork__pert_blocks4bnweight_cluster_fork_args);

    pert_blocks4bnweight_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks4bnweight_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_blocks4bnweight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks4bnweight_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks4bnweight_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks4bnweight_L2_ingress_dma_wait_measurement =
        pert_blocks4bnweight_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks4bnweight_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks4bnweight_L2_kernel_measurement =
        pert_blocks4bnweight_L2_kernel_end_measurements[PROFILING_I] - pert_blocks4bnweight_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks4bnweight_L2_egress_dma_wait_measurement =
        pert_blocks4bnweight_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks4bnweight_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks4bnweight_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks4bnweight_L2_ingress_dma_wait_measurement,
           pert_blocks4bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks4bnweight_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks4bnweight_L2_kernel_measurement,
           pert_blocks4bnweight_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks4bnweight_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks4bnweight_L2_egress_dma_wait_measurement,
           pert_blocks4bnweight_L2_suffix);

    uint32_t total =
        pert_blocks4bnweight_L2_ingress_dma_wait_measurement + pert_blocks4bnweight_L2_kernel_measurement + pert_blocks4bnweight_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks4bnweight_L2_ingress_dma_wait_measurement + pert_blocks4bnweight_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks4bnweight_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks4bnweight_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks4bnweight_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr;
} _pert_blocks4bnweight_closure_L3_args_t;

static void _pert_blocks4bnweight_closure_L3(void *_pert_blocks4bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnweight_closure_L3_args_t *args = (_pert_blocks4bnweight_closure_L3_args_t *)_pert_blocks4bnweight_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor = args->DeeployNetwork_blocks4bnweight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnweight_closure_args_t DeeployNetwork__pert_blocks4bnweight_closure_args = (_pert_blocks4bnweight_closure_args_t){
      .DeeployNetwork_blocks4bnweight_pert_tensor = DeeployNetwork_blocks4bnweight_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr};

  // _pert_blocks4bnweight_closure CLOSURE CALL
  _pert_blocks4bnweight_closure(&DeeployNetwork__pert_blocks4bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref;
} _pert_blocks4bnbias_tiling_closure_args_t;

static void _pert_blocks4bnbias_tiling_closure(void *_pert_blocks4bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_tiling_closure_args_t *args = (_pert_blocks4bnbias_tiling_closure_args_t *)_pert_blocks4bnbias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_blocks4bnbias, Op: PerturbRademacher)
  uint8_t pert_blocks4bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_blocks4bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_blocks4bnbias_chunk = (32 >> pert_blocks4bnbias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_blocks4bnbias_chunk_start = (uint32_t)MIN(pert_blocks4bnbias_chunk * pert_blocks4bnbias_core_id, (uint32_t)32);
  uint32_t pert_blocks4bnbias_chunk_stop = (uint32_t)MIN(pert_blocks4bnbias_chunk_start + pert_blocks4bnbias_chunk, (uint32_t)32);
  uint32_t pert_blocks4bnbias_local_size = pert_blocks4bnbias_chunk_stop - pert_blocks4bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 21 + pert_blocks4bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_blocks4bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref[pert_blocks4bnbias_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref[pert_blocks4bnbias_chunk_start], chunk_seed,
                            perturbation_sign, pert_blocks4bnbias_local_size, zo_eco_mom[21] + pert_blocks4bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref[pert_blocks4bnbias_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref[pert_blocks4bnbias_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_blocks4bnbias_local_size, pert_blocks4bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref[pert_blocks4bnbias_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref[pert_blocks4bnbias_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_blocks4bnbias_local_size, pert_blocks4bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref;
} _pert_blocks4bnbias_cluster_fork_args_t;

static void _pert_blocks4bnbias_cluster_fork(void *_pert_blocks4bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_cluster_fork_args_t *args = (_pert_blocks4bnbias_cluster_fork_args_t *)_pert_blocks4bnbias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_tiling_closure_args_t DeeployNetwork__pert_blocks4bnbias_tiling_closure_args = (_pert_blocks4bnbias_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref};

  // _pert_blocks4bnbias_tiling_closure CLOSURE CALL
  _pert_blocks4bnbias_tiling_closure(&DeeployNetwork__pert_blocks4bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr;
} _pert_blocks4bnbias_closure_args_t;

static void _pert_blocks4bnbias_closure(void *_pert_blocks4bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_closure_args_t *args = (_pert_blocks4bnbias_closure_args_t *)_pert_blocks4bnbias_closure_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor = args->DeeployNetwork_blocks4bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_input_11_ref =
      (void *)((char *)DeeployNetwork_input_11 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_blocks4bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr]);

  const static char pert_blocks4bnbias_L2_suffix[] = " cycles \n";

  const static char pert_blocks4bnbias_L2_prefix[] = "[pert_blocks4bnbias_L2][SB][32 ops][Tile ";

  static uint32_t pert_blocks4bnbias_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks4bnbias_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks4bnbias_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_blocks4bnbias_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_blocks4bnbias_L2_kernel_end_measurements[1];

  static uint32_t pert_blocks4bnbias_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    pert_blocks4bnbias_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_input_11_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_blocks4bnbias_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_blocks4bnbias_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_blocks4bnbias_cluster_fork_args_t DeeployNetwork__pert_blocks4bnbias_cluster_fork_args = (_pert_blocks4bnbias_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_blocks4bnbias_cluster_fork, &DeeployNetwork__pert_blocks4bnbias_cluster_fork_args);

    pert_blocks4bnbias_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_blocks4bnbias_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_blocks4bnbias_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_blocks4bnbias_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_blocks4bnbias_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_blocks4bnbias_L2_ingress_dma_wait_measurement =
        pert_blocks4bnbias_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_blocks4bnbias_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_blocks4bnbias_L2_kernel_measurement =
        pert_blocks4bnbias_L2_kernel_end_measurements[PROFILING_I] - pert_blocks4bnbias_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_blocks4bnbias_L2_egress_dma_wait_measurement =
        pert_blocks4bnbias_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_blocks4bnbias_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_blocks4bnbias_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_blocks4bnbias_L2_ingress_dma_wait_measurement,
           pert_blocks4bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks4bnbias_L2_prefix, PROFILING_I, "Kernel     :", pert_blocks4bnbias_L2_kernel_measurement, pert_blocks4bnbias_L2_suffix);

    printf("%s%u] %s%6u%s", pert_blocks4bnbias_L2_prefix, PROFILING_I, "Post-Kernel:", pert_blocks4bnbias_L2_egress_dma_wait_measurement,
           pert_blocks4bnbias_L2_suffix);

    uint32_t total =
        pert_blocks4bnbias_L2_ingress_dma_wait_measurement + pert_blocks4bnbias_L2_kernel_measurement + pert_blocks4bnbias_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_blocks4bnbias_L2_ingress_dma_wait_measurement + pert_blocks4bnbias_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_blocks4bnbias_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_blocks4bnbias_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_blocks4bnbias_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr;
} _pert_blocks4bnbias_closure_L3_args_t;

static void _pert_blocks4bnbias_closure_L3(void *_pert_blocks4bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_blocks4bnbias_closure_L3_args_t *args = (_pert_blocks4bnbias_closure_L3_args_t *)_pert_blocks4bnbias_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor = args->DeeployNetwork_blocks4bnbias_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_blocks4bnbias_closure_args_t DeeployNetwork__pert_blocks4bnbias_closure_args = (_pert_blocks4bnbias_closure_args_t){
      .DeeployNetwork_blocks4bnbias_pert_tensor = DeeployNetwork_blocks4bnbias_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr};

  // _pert_blocks4bnbias_closure CLOSURE CALL
  _pert_blocks4bnbias_closure(&DeeployNetwork__pert_blocks4bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref;
} _pert_fcweight_fp32_tiling_closure_args_t;

static void _pert_fcweight_fp32_tiling_closure(void *_pert_fcweight_fp32_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_tiling_closure_args_t *args = (_pert_fcweight_fp32_tiling_closure_args_t *)_pert_fcweight_fp32_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_fcweight_fp32, Op: PerturbRademacher)
  uint8_t pert_fcweight_fp32_core_id = (uint8_t)pi_core_id();
  uint8_t pert_fcweight_fp32_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_fcweight_fp32_chunk = (288 >> pert_fcweight_fp32_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_fcweight_fp32_chunk_start = (uint32_t)MIN(pert_fcweight_fp32_chunk * pert_fcweight_fp32_core_id, (uint32_t)288);
  uint32_t pert_fcweight_fp32_chunk_stop = (uint32_t)MIN(pert_fcweight_fp32_chunk_start + pert_fcweight_fp32_chunk, (uint32_t)288);
  uint32_t pert_fcweight_fp32_local_size = pert_fcweight_fp32_chunk_stop - pert_fcweight_fp32_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_fcweight_fp32_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_fcweight_fp32_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref[pert_fcweight_fp32_chunk_start],
                            (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref[pert_fcweight_fp32_chunk_start], chunk_seed,
                            perturbation_sign, pert_fcweight_fp32_local_size, zo_eco_mom[10] + pert_fcweight_fp32_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref[pert_fcweight_fp32_chunk_start],
                                          (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref[pert_fcweight_fp32_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          pert_fcweight_fp32_local_size, pert_fcweight_fp32_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref[pert_fcweight_fp32_chunk_start],
                                (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref[pert_fcweight_fp32_chunk_start], chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_fcweight_fp32_local_size, pert_fcweight_fp32_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref;
} _pert_fcweight_fp32_cluster_fork_args_t;

static void _pert_fcweight_fp32_cluster_fork(void *_pert_fcweight_fp32_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_cluster_fork_args_t *args = (_pert_fcweight_fp32_cluster_fork_args_t *)_pert_fcweight_fp32_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_tiling_closure_args_t DeeployNetwork__pert_fcweight_fp32_tiling_closure_args = (_pert_fcweight_fp32_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref};

  // _pert_fcweight_fp32_tiling_closure CLOSURE CALL
  _pert_fcweight_fp32_tiling_closure(&DeeployNetwork__pert_fcweight_fp32_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr;
} _pert_fcweight_fp32_closure_args_t;

static void _pert_fcweight_fp32_closure(void *_pert_fcweight_fp32_closure_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_closure_args_t *args = (_pert_fcweight_fp32_closure_args_t *)_pert_fcweight_fp32_closure_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor = args->DeeployNetwork_fcweight_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_input_22_ref =
      (void *)((char *)DeeployNetwork_input_22 +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_fcweight_fp32_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr]);

  const static char pert_fcweight_fp32_L2_suffix[] = " cycles \n";

  const static char pert_fcweight_fp32_L2_prefix[] = "[pert_fcweight_fp32_L2][SB][288 ops][Tile ";

  static uint32_t pert_fcweight_fp32_L2_egress_dma_wait_end_measurements[1];

  static uint32_t pert_fcweight_fp32_L2_egress_dma_wait_start_measurements[1];

  static uint32_t pert_fcweight_fp32_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t pert_fcweight_fp32_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t pert_fcweight_fp32_L2_kernel_end_measurements[1];

  static uint32_t pert_fcweight_fp32_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr) + 1];
       TILING_I++) {

    pert_fcweight_fp32_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_input_22_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    pert_fcweight_fp32_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    pert_fcweight_fp32_L2_kernel_start_measurements[TILING_I] = getCycles();

    _pert_fcweight_fp32_cluster_fork_args_t DeeployNetwork__pert_fcweight_fp32_cluster_fork_args = (_pert_fcweight_fp32_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_fcweight_fp32_cluster_fork, &DeeployNetwork__pert_fcweight_fp32_cluster_fork_args);

    pert_fcweight_fp32_L2_kernel_end_measurements[TILING_I] = getCycles();

    pert_fcweight_fp32_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_fcweight_fp32_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    pert_fcweight_fp32_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling pert_fcweight_fp32_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr];
       PROFILING_I++) {

    uint32_t pert_fcweight_fp32_L2_ingress_dma_wait_measurement =
        pert_fcweight_fp32_L2_ingress_dma_wait_end_measurements[PROFILING_I] - pert_fcweight_fp32_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t pert_fcweight_fp32_L2_kernel_measurement =
        pert_fcweight_fp32_L2_kernel_end_measurements[PROFILING_I] - pert_fcweight_fp32_L2_kernel_start_measurements[PROFILING_I];

    uint32_t pert_fcweight_fp32_L2_egress_dma_wait_measurement =
        pert_fcweight_fp32_L2_egress_dma_wait_end_measurements[PROFILING_I] - pert_fcweight_fp32_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", pert_fcweight_fp32_L2_prefix, PROFILING_I, "Pre-Kernel :", pert_fcweight_fp32_L2_ingress_dma_wait_measurement,
           pert_fcweight_fp32_L2_suffix);

    printf("%s%u] %s%6u%s", pert_fcweight_fp32_L2_prefix, PROFILING_I, "Kernel     :", pert_fcweight_fp32_L2_kernel_measurement, pert_fcweight_fp32_L2_suffix);

    printf("%s%u] %s%6u%s", pert_fcweight_fp32_L2_prefix, PROFILING_I, "Post-Kernel:", pert_fcweight_fp32_L2_egress_dma_wait_measurement,
           pert_fcweight_fp32_L2_suffix);

    uint32_t total =
        pert_fcweight_fp32_L2_ingress_dma_wait_measurement + pert_fcweight_fp32_L2_kernel_measurement + pert_fcweight_fp32_L2_egress_dma_wait_measurement;
    uint32_t dma = pert_fcweight_fp32_L2_ingress_dma_wait_measurement + pert_fcweight_fp32_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : pert_fcweight_fp32_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", pert_fcweight_fp32_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, pert_fcweight_fp32_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr;
} _pert_fcweight_fp32_closure_L3_args_t;

static void _pert_fcweight_fp32_closure_L3(void *_pert_fcweight_fp32_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_fcweight_fp32_closure_L3_args_t *args = (_pert_fcweight_fp32_closure_L3_args_t *)_pert_fcweight_fp32_closure_L3_args;

  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor = args->DeeployNetwork_fcweight_fp32_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_fcweight_fp32_closure_args_t DeeployNetwork__pert_fcweight_fp32_closure_args = (_pert_fcweight_fp32_closure_args_t){
      .DeeployNetwork_fcweight_fp32_pert_tensor = DeeployNetwork_fcweight_fp32_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr};

  // _pert_fcweight_fp32_closure CLOSURE CALL
  _pert_fcweight_fp32_closure(&DeeployNetwork__pert_fcweight_fp32_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;
} _rqsp_blocks1convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks1convweight_int8_tiling_closure(void *_rqsp_blocks1convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks1convweight_int8_tiling_closure_args_t *)_rqsp_blocks1convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks1convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks1convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks1convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks1convweight_int8_chunk = (2048 >> rqsp_blocks1convweight_int8_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks1convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks1convweight_int8_chunk * rqsp_blocks1convweight_int8_core_id, (uint32_t)2048);
  uint32_t rqsp_blocks1convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks1convweight_int8_chunk_start + rqsp_blocks1convweight_int8_chunk, (uint32_t)2048);
  uint32_t rqsp_blocks1convweight_int8_local_size = rqsp_blocks1convweight_int8_chunk_stop - rqsp_blocks1convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks1convweight_int8_channel_start_offset = rqsp_blocks1convweight_int8_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 2 + rqsp_blocks1convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks1convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref[rqsp_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref[rqsp_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks1convweight_int8_local_size,
        rqsp_blocks1convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[2] + rqsp_blocks1convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref[rqsp_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref[rqsp_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks1convweight_int8_local_size,
        rqsp_blocks1convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks1convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;
} _rqsp_blocks1convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks1convweight_int8_cluster_fork(void *_rqsp_blocks1convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks1convweight_int8_cluster_fork_args_t *)_rqsp_blocks1convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks1convweight_int8_tiling_closure_args =
      (_rqsp_blocks1convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref};

  // _rqsp_blocks1convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks1convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks1convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
} _rqsp_blocks1convweight_int8_closure_args_t;

static void _rqsp_blocks1convweight_int8_closure(void *_rqsp_blocks1convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_closure_args_t *args = (_rqsp_blocks1convweight_int8_closure_args_t *)_rqsp_blocks1convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_input_13_ref =
      (void *)((char *)DeeployNetwork_input_13 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);

  const static char rqsp_blocks1convweight_int8_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks1convweight_int8_L2_prefix[] = "[rqsp_blocks1convweight_int8_L2][SB][2048 ops][Tile ";

  static uint32_t rqsp_blocks1convweight_int8_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks1convweight_int8_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks1convweight_int8_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks1convweight_int8_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks1convweight_int8_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks1convweight_int8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks1convweight_int8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_input_13_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks1convweight_int8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks1convweight_int8_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks1convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks1convweight_int8_cluster_fork_args =
        (_rqsp_blocks1convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks1convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks1convweight_int8_cluster_fork_args);

    rqsp_blocks1convweight_int8_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks1convweight_int8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks1convweight_int8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks1convweight_int8_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks1convweight_int8_L2_ingress_dma_wait_measurement = rqsp_blocks1convweight_int8_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks1convweight_int8_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks1convweight_int8_L2_kernel_measurement =
        rqsp_blocks1convweight_int8_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks1convweight_int8_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks1convweight_int8_L2_egress_dma_wait_measurement = rqsp_blocks1convweight_int8_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks1convweight_int8_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks1convweight_int8_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks1convweight_int8_L2_ingress_dma_wait_measurement,
           rqsp_blocks1convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks1convweight_int8_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks1convweight_int8_L2_kernel_measurement,
           rqsp_blocks1convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks1convweight_int8_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks1convweight_int8_L2_egress_dma_wait_measurement,
           rqsp_blocks1convweight_int8_L2_suffix);

    uint32_t total = rqsp_blocks1convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks1convweight_int8_L2_kernel_measurement +
                     rqsp_blocks1convweight_int8_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks1convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks1convweight_int8_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks1convweight_int8_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks1convweight_int8_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks1convweight_int8_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
} _rqsp_blocks1convweight_int8_closure_L3_args_t;

static void _rqsp_blocks1convweight_int8_closure_L3(void *_rqsp_blocks1convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_closure_L3_args_t *args = (_rqsp_blocks1convweight_int8_closure_L3_args_t *)_rqsp_blocks1convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks1convweight_int8_closure_args = (_rqsp_blocks1convweight_int8_closure_args_t){
      .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr};

  // _rqsp_blocks1convweight_int8_closure CLOSURE CALL
  _rqsp_blocks1convweight_int8_closure(&DeeployNetwork__rqsp_blocks1convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [16, 8, 1, 16] -> [16, 1, 16, 8] (Name: _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 16;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 16;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_blocks1convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443840,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_blocks1convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312768,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref;
} _rqsp_blocks2convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks2convweight_int8_tiling_closure(void *_rqsp_blocks2convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks2convweight_int8_tiling_closure_args_t *)_rqsp_blocks2convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks2convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks2convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks2convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks2convweight_int8_chunk = (2048 >> rqsp_blocks2convweight_int8_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks2convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks2convweight_int8_chunk * rqsp_blocks2convweight_int8_core_id, (uint32_t)2048);
  uint32_t rqsp_blocks2convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks2convweight_int8_chunk_start + rqsp_blocks2convweight_int8_chunk, (uint32_t)2048);
  uint32_t rqsp_blocks2convweight_int8_local_size = rqsp_blocks2convweight_int8_chunk_stop - rqsp_blocks2convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks2convweight_int8_channel_start_offset = rqsp_blocks2convweight_int8_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 4 + rqsp_blocks2convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks2convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref[rqsp_blocks2convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref[rqsp_blocks2convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks2convweight_int8_local_size,
        rqsp_blocks2convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[4] + rqsp_blocks2convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref[rqsp_blocks2convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref[rqsp_blocks2convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks2convweight_int8_local_size,
        rqsp_blocks2convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks2convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref;
} _rqsp_blocks2convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks2convweight_int8_cluster_fork(void *_rqsp_blocks2convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks2convweight_int8_cluster_fork_args_t *)_rqsp_blocks2convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks2convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks2convweight_int8_tiling_closure_args =
      (_rqsp_blocks2convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref};

  // _rqsp_blocks2convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks2convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks2convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr;
} _rqsp_blocks2convweight_int8_closure_args_t;

static void _rqsp_blocks2convweight_int8_closure(void *_rqsp_blocks2convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convweight_int8_closure_args_t *args = (_rqsp_blocks2convweight_int8_closure_args_t *)_rqsp_blocks2convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor = args->DeeployNetwork_blocks2convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_input_14_ref =
      (void *)((char *)DeeployNetwork_input_14 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_blocks2convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_blocks2convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr]);

  const static char rqsp_blocks2convweight_int8_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks2convweight_int8_L2_prefix[] = "[rqsp_blocks2convweight_int8_L2][SB][2048 ops][Tile ";

  static uint32_t rqsp_blocks2convweight_int8_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks2convweight_int8_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks2convweight_int8_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks2convweight_int8_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks2convweight_int8_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks2convweight_int8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks2convweight_int8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_input_14_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_blocks2convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks2convweight_int8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks2convweight_int8_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks2convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks2convweight_int8_cluster_fork_args =
        (_rqsp_blocks2convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks2convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks2convweight_int8_cluster_fork_args);

    rqsp_blocks2convweight_int8_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks2convweight_int8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_blocks2convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks2convweight_int8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks2convweight_int8_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks2convweight_int8_L2_ingress_dma_wait_measurement = rqsp_blocks2convweight_int8_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks2convweight_int8_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks2convweight_int8_L2_kernel_measurement =
        rqsp_blocks2convweight_int8_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks2convweight_int8_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks2convweight_int8_L2_egress_dma_wait_measurement = rqsp_blocks2convweight_int8_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks2convweight_int8_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks2convweight_int8_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks2convweight_int8_L2_ingress_dma_wait_measurement,
           rqsp_blocks2convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks2convweight_int8_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks2convweight_int8_L2_kernel_measurement,
           rqsp_blocks2convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks2convweight_int8_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks2convweight_int8_L2_egress_dma_wait_measurement,
           rqsp_blocks2convweight_int8_L2_suffix);

    uint32_t total = rqsp_blocks2convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks2convweight_int8_L2_kernel_measurement +
                     rqsp_blocks2convweight_int8_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks2convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks2convweight_int8_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks2convweight_int8_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks2convweight_int8_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks2convweight_int8_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr;
} _rqsp_blocks2convweight_int8_closure_L3_args_t;

static void _rqsp_blocks2convweight_int8_closure_L3(void *_rqsp_blocks2convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks2convweight_int8_closure_L3_args_t *args = (_rqsp_blocks2convweight_int8_closure_L3_args_t *)_rqsp_blocks2convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor = args->DeeployNetwork_blocks2convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks2convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks2convweight_int8_closure_args = (_rqsp_blocks2convweight_int8_closure_args_t){
      .DeeployNetwork_blocks2convweight_int8_pert_tensor = DeeployNetwork_blocks2convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr};

  // _rqsp_blocks2convweight_int8_closure CLOSURE CALL
  _rqsp_blocks2convweight_int8_closure(&DeeployNetwork__rqsp_blocks2convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [16, 16, 1, 8] -> [16, 1, 8, 16] (Name: _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 16;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 8;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor = args->DeeployNetwork_blocks2convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_blocks2convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443840,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_blocks2convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312768,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor = args->DeeployNetwork_blocks2convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks2convweight_int8_pert_tensor = DeeployNetwork_blocks2convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;
} _rqsp_blocks3convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks3convweight_int8_tiling_closure(void *_rqsp_blocks3convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks3convweight_int8_tiling_closure_args_t *)_rqsp_blocks3convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks3convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks3convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks3convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks3convweight_int8_chunk = (3584 >> rqsp_blocks3convweight_int8_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks3convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks3convweight_int8_chunk * rqsp_blocks3convweight_int8_core_id, (uint32_t)3584);
  uint32_t rqsp_blocks3convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks3convweight_int8_chunk_start + rqsp_blocks3convweight_int8_chunk, (uint32_t)3584);
  uint32_t rqsp_blocks3convweight_int8_local_size = rqsp_blocks3convweight_int8_chunk_stop - rqsp_blocks3convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks3convweight_int8_channel_start_offset = rqsp_blocks3convweight_int8_chunk_start % 112;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 6 + rqsp_blocks3convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks3convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref[rqsp_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref[rqsp_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed, rqsp_blocks3convweight_int8_local_size,
        rqsp_blocks3convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[6] + rqsp_blocks3convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref[rqsp_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref[rqsp_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed, rqsp_blocks3convweight_int8_local_size,
        rqsp_blocks3convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks3convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;
} _rqsp_blocks3convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks3convweight_int8_cluster_fork(void *_rqsp_blocks3convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks3convweight_int8_cluster_fork_args_t *)_rqsp_blocks3convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks3convweight_int8_tiling_closure_args =
      (_rqsp_blocks3convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref};

  // _rqsp_blocks3convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks3convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks3convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
} _rqsp_blocks3convweight_int8_closure_args_t;

static void _rqsp_blocks3convweight_int8_closure(void *_rqsp_blocks3convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_closure_args_t *args = (_rqsp_blocks3convweight_int8_closure_args_t *)_rqsp_blocks3convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_input_15_ref =
      (void *)((char *)DeeployNetwork_input_15 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);

  const static char rqsp_blocks3convweight_int8_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks3convweight_int8_L2_prefix[] = "[rqsp_blocks3convweight_int8_L2][SB][3584 ops][Tile ";

  static uint32_t rqsp_blocks3convweight_int8_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks3convweight_int8_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks3convweight_int8_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks3convweight_int8_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks3convweight_int8_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks3convweight_int8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks3convweight_int8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_input_15_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks3convweight_int8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks3convweight_int8_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks3convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks3convweight_int8_cluster_fork_args =
        (_rqsp_blocks3convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks3convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks3convweight_int8_cluster_fork_args);

    rqsp_blocks3convweight_int8_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks3convweight_int8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks3convweight_int8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks3convweight_int8_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks3convweight_int8_L2_ingress_dma_wait_measurement = rqsp_blocks3convweight_int8_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks3convweight_int8_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks3convweight_int8_L2_kernel_measurement =
        rqsp_blocks3convweight_int8_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks3convweight_int8_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks3convweight_int8_L2_egress_dma_wait_measurement = rqsp_blocks3convweight_int8_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks3convweight_int8_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks3convweight_int8_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks3convweight_int8_L2_ingress_dma_wait_measurement,
           rqsp_blocks3convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks3convweight_int8_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks3convweight_int8_L2_kernel_measurement,
           rqsp_blocks3convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks3convweight_int8_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks3convweight_int8_L2_egress_dma_wait_measurement,
           rqsp_blocks3convweight_int8_L2_suffix);

    uint32_t total = rqsp_blocks3convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks3convweight_int8_L2_kernel_measurement +
                     rqsp_blocks3convweight_int8_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks3convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks3convweight_int8_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks3convweight_int8_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks3convweight_int8_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks3convweight_int8_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
} _rqsp_blocks3convweight_int8_closure_L3_args_t;

static void _rqsp_blocks3convweight_int8_closure_L3(void *_rqsp_blocks3convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_closure_L3_args_t *args = (_rqsp_blocks3convweight_int8_closure_L3_args_t *)_rqsp_blocks3convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks3convweight_int8_closure_args = (_rqsp_blocks3convweight_int8_closure_args_t){
      .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr};

  // _rqsp_blocks3convweight_int8_closure CLOSURE CALL
  _rqsp_blocks3convweight_int8_closure(&DeeployNetwork__rqsp_blocks3convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [32, 16, 7, 1] -> [32, 7, 1, 16] (Name: _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 32;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 7;

  uint16_t dimLen_3 = 1;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_blocks3convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445376,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_blocks3convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1314304,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref;
} _rqsp_blocks4convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks4convweight_int8_tiling_closure(void *_rqsp_blocks4convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks4convweight_int8_tiling_closure_args_t *)_rqsp_blocks4convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks4convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks4convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks4convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks4convweight_int8_chunk = (7168 >> rqsp_blocks4convweight_int8_log2Core) + ((7168 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks4convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks4convweight_int8_chunk * rqsp_blocks4convweight_int8_core_id, (uint32_t)7168);
  uint32_t rqsp_blocks4convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks4convweight_int8_chunk_start + rqsp_blocks4convweight_int8_chunk, (uint32_t)7168);
  uint32_t rqsp_blocks4convweight_int8_local_size = rqsp_blocks4convweight_int8_chunk_stop - rqsp_blocks4convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks4convweight_int8_channel_start_offset = rqsp_blocks4convweight_int8_chunk_start % 224;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 8 + rqsp_blocks4convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks4convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref[rqsp_blocks4convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref[rqsp_blocks4convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref, 15, 224, chunk_seed, rqsp_blocks4convweight_int8_local_size,
        rqsp_blocks4convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[8] + rqsp_blocks4convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref[rqsp_blocks4convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref[rqsp_blocks4convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref, 15, 224, chunk_seed, rqsp_blocks4convweight_int8_local_size,
        rqsp_blocks4convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks4convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref;
} _rqsp_blocks4convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks4convweight_int8_cluster_fork(void *_rqsp_blocks4convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks4convweight_int8_cluster_fork_args_t *)_rqsp_blocks4convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks4convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks4convweight_int8_tiling_closure_args =
      (_rqsp_blocks4convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref};

  // _rqsp_blocks4convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks4convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks4convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr;
} _rqsp_blocks4convweight_int8_closure_args_t;

static void _rqsp_blocks4convweight_int8_closure(void *_rqsp_blocks4convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convweight_int8_closure_args_t *args = (_rqsp_blocks4convweight_int8_closure_args_t *)_rqsp_blocks4convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor = args->DeeployNetwork_blocks4convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_input_16_ref =
      (void *)((char *)DeeployNetwork_input_16 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_blocks4convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr]);

  const static char rqsp_blocks4convweight_int8_L2_suffix[] = " cycles \n";

  const static char rqsp_blocks4convweight_int8_L2_prefix[] = "[rqsp_blocks4convweight_int8_L2][SB][7168 ops][Tile ";

  static uint32_t rqsp_blocks4convweight_int8_L2_egress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks4convweight_int8_L2_egress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks4convweight_int8_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t rqsp_blocks4convweight_int8_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t rqsp_blocks4convweight_int8_L2_kernel_end_measurements[1];

  static uint32_t rqsp_blocks4convweight_int8_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    rqsp_blocks4convweight_int8_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_input_16_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_blocks4convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    rqsp_blocks4convweight_int8_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    rqsp_blocks4convweight_int8_L2_kernel_start_measurements[TILING_I] = getCycles();

    _rqsp_blocks4convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks4convweight_int8_cluster_fork_args =
        (_rqsp_blocks4convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks4convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks4convweight_int8_cluster_fork_args);

    rqsp_blocks4convweight_int8_L2_kernel_end_measurements[TILING_I] = getCycles();

    rqsp_blocks4convweight_int8_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_blocks4convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    rqsp_blocks4convweight_int8_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling rqsp_blocks4convweight_int8_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr];
       PROFILING_I++) {

    uint32_t rqsp_blocks4convweight_int8_L2_ingress_dma_wait_measurement = rqsp_blocks4convweight_int8_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           rqsp_blocks4convweight_int8_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks4convweight_int8_L2_kernel_measurement =
        rqsp_blocks4convweight_int8_L2_kernel_end_measurements[PROFILING_I] - rqsp_blocks4convweight_int8_L2_kernel_start_measurements[PROFILING_I];

    uint32_t rqsp_blocks4convweight_int8_L2_egress_dma_wait_measurement = rqsp_blocks4convweight_int8_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          rqsp_blocks4convweight_int8_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", rqsp_blocks4convweight_int8_L2_prefix, PROFILING_I, "Pre-Kernel :", rqsp_blocks4convweight_int8_L2_ingress_dma_wait_measurement,
           rqsp_blocks4convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks4convweight_int8_L2_prefix, PROFILING_I, "Kernel     :", rqsp_blocks4convweight_int8_L2_kernel_measurement,
           rqsp_blocks4convweight_int8_L2_suffix);

    printf("%s%u] %s%6u%s", rqsp_blocks4convweight_int8_L2_prefix, PROFILING_I, "Post-Kernel:", rqsp_blocks4convweight_int8_L2_egress_dma_wait_measurement,
           rqsp_blocks4convweight_int8_L2_suffix);

    uint32_t total = rqsp_blocks4convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks4convweight_int8_L2_kernel_measurement +
                     rqsp_blocks4convweight_int8_L2_egress_dma_wait_measurement;
    uint32_t dma = rqsp_blocks4convweight_int8_L2_ingress_dma_wait_measurement + rqsp_blocks4convweight_int8_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : rqsp_blocks4convweight_int8_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", rqsp_blocks4convweight_int8_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, rqsp_blocks4convweight_int8_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr;
} _rqsp_blocks4convweight_int8_closure_L3_args_t;

static void _rqsp_blocks4convweight_int8_closure_L3(void *_rqsp_blocks4convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks4convweight_int8_closure_L3_args_t *args = (_rqsp_blocks4convweight_int8_closure_L3_args_t *)_rqsp_blocks4convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor = args->DeeployNetwork_blocks4convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks4convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks4convweight_int8_closure_args = (_rqsp_blocks4convweight_int8_closure_args_t){
      .DeeployNetwork_blocks4convweight_int8_pert_tensor = DeeployNetwork_blocks4convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr};

  // _rqsp_blocks4convweight_int8_closure CLOSURE CALL
  _rqsp_blocks4convweight_int8_closure(&DeeployNetwork__rqsp_blocks4convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [32, 32, 7, 1] -> [32, 7, 1, 32] (Name: _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 32;

  uint16_t dimLen_1 = 32;

  uint16_t dimLen_2 = 7;

  uint16_t dimLen_3 = 1;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor = args->DeeployNetwork_blocks4convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_blocks4convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1448960,
          .size = 7168,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_blocks4convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1317888,
          .size = 7168,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor = args->DeeployNetwork_blocks4convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks4convweight_int8_pert_tensor = DeeployNetwork_blocks4convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_args);

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

  const static char QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_suffix[] = " cycles \n";

  const static char QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_prefix[] = "[QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2][SB][0 ops][Tile ";

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_end_measurements[1];

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_start_measurements[1];

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_end_measurements[1];

  static uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr) + 1];
       TILING_I++) {

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1480992, DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_start_measurements[TILING_I] = getCycles();

    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args =
        (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork,
                    &DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_cluster_fork_args);

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_end_measurements[TILING_I] = getCycles();

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1320520, DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr];
       PROFILING_I++) {

    uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_measurement =
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_measurement =
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_end_measurements[PROFILING_I] -
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_start_measurements[PROFILING_I];

    uint32_t QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_measurement =
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_prefix, PROFILING_I,
           "Pre-Kernel :", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_measurement, QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_suffix);

    printf("%s%u] %s%6u%s", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_prefix, PROFILING_I,
           "Kernel     :", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_measurement, QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_suffix);

    printf("%s%u] %s%6u%s", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_prefix, PROFILING_I,
           "Post-Kernel:", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_measurement, QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_suffix);

    uint32_t total = QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_measurement +
                     QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_measurement +
                     QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_measurement;
    uint32_t dma =
        QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_ingress_dma_wait_measurement + QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_prefix, PROFILING_I,
           total, kernel_percentage, overhead_percentage, QCDQ_blocks_0_conv_input_quant_1Clip_Quant_L2_kernel_measurement, dma);
  }
  StartTimer();

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 1, 14, 700] -> [1, 14, 700, 1] (Name: _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 700;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src =
      (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9800);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref =
          (void *)((char *)DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr]);

  const static char _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2][SB][0 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1451592,
          .size = 9800,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_blocks_0_conv_input_quant_1Clip_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1320520,
          .size = 9800,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles[(
                      *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_suffix);

    uint32_t total = _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_measurement +
                     _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_measurement +
                     _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_ingress_dma_wait_measurement +
                   _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage =
        (total == 0) ? 0 : _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           _MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = args->DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_tiling_closure(void *__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 16,
                        700, 14, 1, 701, 14, 8, 4, 1, 0, 0, 2, 2, 1, 1, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_cluster_fork(void *__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 78512);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88376);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88440);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88408);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88312);

  const static char _MERGE_CONVRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_L2][SB][657538 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1451592,
          .size = 9800,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441824,
          .size = 32,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441824,
          .size = 32,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_0_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_cluster_fork_args = (__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_0_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1389232,
          .size = 78512,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_0_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Pre-Kernel :", _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel     :", _MERGE_CONVRQ_PASS_0_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Post-Kernel:", _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    uint32_t total =
        _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_0_L2_kernel_measurement + _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_0_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, _MERGE_CONVRQ_PASS_0_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
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
      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref, 14,
      701, 87, 8, 1, 8, 9.999999747378752e-06f, (float32_t)6.1015625, (float32_t)0.0, (float32_t)8.295696202531646, (float32_t)0.0, -128, 127);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
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
  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_args_t;

static void _bnBatchNormalization_tail_closure(void *_bnBatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_args_t *args = (_bnBatchNormalization_tail_closure_args_t *)_bnBatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor = args->DeeployNetwork_blocks0bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor = args->DeeployNetwork_blocks0bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88288);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88256);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88320);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88352);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 78512);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnweight_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnbias_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_cumByteOffset
                                                                      [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_mean_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_cumByteOffset
                                                                         [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0bnrunning_var_tensor + DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr]);

  const static char bnBatchNormalization_tail_L2_suffix[] = " cycles \n";

  const static char bnBatchNormalization_tail_L2_prefix[] = "[bnBatchNormalization_tail_L2][SB][77952 ops][Tile ";

  static uint32_t bnBatchNormalization_tail_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bnBatchNormalization_tail_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bnBatchNormalization_tail_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bnBatchNormalization_tail_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bnBatchNormalization_tail_L2_kernel_end_measurements[1];

  static uint32_t bnBatchNormalization_tail_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    bnBatchNormalization_tail_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1520304, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnweight_pert_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnbias_pert_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_mean_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_blocks0bnrunning_var_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    bnBatchNormalization_tail_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bnBatchNormalization_tail_L2_kernel_start_measurements[TILING_I] = getCycles();

    _bnBatchNormalization_tail_cluster_fork_args_t DeeployNetwork__bnBatchNormalization_tail_cluster_fork_args =
        (_bnBatchNormalization_tail_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_scale_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_bias_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_mean_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref =
                DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_running_var_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bnBatchNormalization_tail_cluster_fork, &DeeployNetwork__bnBatchNormalization_tail_cluster_fork_args);

    bnBatchNormalization_tail_L2_kernel_end_measurements[TILING_I] = getCycles();

    bnBatchNormalization_tail_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1320464, DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    bnBatchNormalization_tail_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bnBatchNormalization_tail_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bnBatchNormalization_tail_L2_ingress_dma_wait_measurement = bnBatchNormalization_tail_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                         bnBatchNormalization_tail_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bnBatchNormalization_tail_L2_kernel_measurement =
        bnBatchNormalization_tail_L2_kernel_end_measurements[PROFILING_I] - bnBatchNormalization_tail_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bnBatchNormalization_tail_L2_egress_dma_wait_measurement = bnBatchNormalization_tail_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                        bnBatchNormalization_tail_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bnBatchNormalization_tail_L2_prefix, PROFILING_I, "Pre-Kernel :", bnBatchNormalization_tail_L2_ingress_dma_wait_measurement,
           bnBatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bnBatchNormalization_tail_L2_prefix, PROFILING_I, "Kernel     :", bnBatchNormalization_tail_L2_kernel_measurement,
           bnBatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bnBatchNormalization_tail_L2_prefix, PROFILING_I, "Post-Kernel:", bnBatchNormalization_tail_L2_egress_dma_wait_measurement,
           bnBatchNormalization_tail_L2_suffix);

    uint32_t total = bnBatchNormalization_tail_L2_ingress_dma_wait_measurement + bnBatchNormalization_tail_L2_kernel_measurement +
                     bnBatchNormalization_tail_L2_egress_dma_wait_measurement;
    uint32_t dma = bnBatchNormalization_tail_L2_ingress_dma_wait_measurement + bnBatchNormalization_tail_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : bnBatchNormalization_tail_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", bnBatchNormalization_tail_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, bnBatchNormalization_tail_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
} _bnBatchNormalization_tail_closure_L3_args_t;

static void _bnBatchNormalization_tail_closure_L3(void *_bnBatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bnBatchNormalization_tail_closure_L3_args_t *args = (_bnBatchNormalization_tail_closure_L3_args_t *)_bnBatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor = args->DeeployNetwork_blocks0bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor = args->DeeployNetwork_blocks0bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bnBatchNormalization_tail_closure_args_t DeeployNetwork__bnBatchNormalization_tail_closure_args = (_bnBatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks0bnweight_pert_tensor = DeeployNetwork_blocks0bnweight_pert_tensor,
      .DeeployNetwork_blocks0bnbias_pert_tensor = DeeployNetwork_blocks0bnbias_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

  // _bnBatchNormalization_tail_closure CLOSURE CALL
  _bnBatchNormalization_tail_closure(&DeeployNetwork__bnBatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} __MERGE_CONVRQ_PASS_1_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_tiling_closure(void *__MERGE_CONVRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_1_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref, 1, 16, 87,
                        14, 8, 88, 14, 16, 16, 1, 0, 0, 8, 8, 1, 1, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} __MERGE_CONVRQ_PASS_1_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_1_cluster_fork(void *__MERGE_CONVRQ_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_1_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_1_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_1_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_tiling_closure_args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

  // __MERGE_CONVRQ_PASS_1_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_1_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_closure(void *__MERGE_CONVRQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_closure_args_t *args = (__MERGE_CONVRQ_PASS_1_closure_args_t *)__MERGE_CONVRQ_PASS_1_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29456);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 33616);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 33552);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31504);

  const static char _MERGE_CONVRQ_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_1_L2_prefix[] = "[_MERGE_CONVRQ_PASS_1_L2][SB][5049968 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1451536,
          .size = 9744,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443840,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_1_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_cluster_fork_args = (__MERGE_CONVRQ_PASS_1_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_1_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_1_cluster_fork_args);

    _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1330432,
          .size = 19712,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_1_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Pre-Kernel :", _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Kernel     :", _MERGE_CONVRQ_PASS_1_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Post-Kernel:", _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    uint32_t total =
        _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_1_L2_kernel_measurement + _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_1_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, _MERGE_CONVRQ_PASS_1_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_1_closure_L3(void *__MERGE_CONVRQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_1_closure_L3_args_t *)__MERGE_CONVRQ_PASS_1_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_1_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_args = (__MERGE_CONVRQ_PASS_1_closure_args_t){
      .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_1_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_1_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_args);

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
      14, 88, 22, 16, 1, 4, 9.999999747378752e-06f, (float32_t)0.1591796875, (float32_t)0.0, (float32_t)15.398496240601503, (float32_t)0.0, -128, 127);

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
  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_args_t;

static void _bn_1BatchNormalization_tail_closure(void *_bn_1BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_args_t *args = (_bn_1BatchNormalization_tail_closure_args_t *)_bn_1BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor = args->DeeployNetwork_blocks1bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor = args->DeeployNetwork_blocks1bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 24704);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 24640);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 24768);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 24832);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnweight_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnbias_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_cumByteOffset
                                                                      [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_mean_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_cumByteOffset
                                                                         [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1bnrunning_var_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr]);

  const static char bn_1BatchNormalization_tail_L2_suffix[] = " cycles \n";

  const static char bn_1BatchNormalization_tail_L2_prefix[] = "[bn_1BatchNormalization_tail_L2][SB][19712 ops][Tile ";

  static uint32_t bn_1BatchNormalization_tail_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bn_1BatchNormalization_tail_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bn_1BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bn_1BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bn_1BatchNormalization_tail_L2_kernel_end_measurements[1];

  static uint32_t bn_1BatchNormalization_tail_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    bn_1BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1461504, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnweight_pert_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnbias_pert_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_mean_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_blocks1bnrunning_var_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    bn_1BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bn_1BatchNormalization_tail_L2_kernel_start_measurements[TILING_I] = getCycles();

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

    bn_1BatchNormalization_tail_L2_kernel_end_measurements[TILING_I] = getCycles();

    bn_1BatchNormalization_tail_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1315648, DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    bn_1BatchNormalization_tail_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bn_1BatchNormalization_tail_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bn_1BatchNormalization_tail_L2_ingress_dma_wait_measurement = bn_1BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           bn_1BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bn_1BatchNormalization_tail_L2_kernel_measurement =
        bn_1BatchNormalization_tail_L2_kernel_end_measurements[PROFILING_I] - bn_1BatchNormalization_tail_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bn_1BatchNormalization_tail_L2_egress_dma_wait_measurement = bn_1BatchNormalization_tail_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          bn_1BatchNormalization_tail_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bn_1BatchNormalization_tail_L2_prefix, PROFILING_I, "Pre-Kernel :", bn_1BatchNormalization_tail_L2_ingress_dma_wait_measurement,
           bn_1BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_1BatchNormalization_tail_L2_prefix, PROFILING_I, "Kernel     :", bn_1BatchNormalization_tail_L2_kernel_measurement,
           bn_1BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_1BatchNormalization_tail_L2_prefix, PROFILING_I, "Post-Kernel:", bn_1BatchNormalization_tail_L2_egress_dma_wait_measurement,
           bn_1BatchNormalization_tail_L2_suffix);

    uint32_t total = bn_1BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_1BatchNormalization_tail_L2_kernel_measurement +
                     bn_1BatchNormalization_tail_L2_egress_dma_wait_measurement;
    uint32_t dma = bn_1BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_1BatchNormalization_tail_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : bn_1BatchNormalization_tail_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", bn_1BatchNormalization_tail_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, bn_1BatchNormalization_tail_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
} _bn_1BatchNormalization_tail_closure_L3_args_t;

static void _bn_1BatchNormalization_tail_closure_L3(void *_bn_1BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_1BatchNormalization_tail_closure_L3_args_t *args = (_bn_1BatchNormalization_tail_closure_L3_args_t *)_bn_1BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor = args->DeeployNetwork_blocks1bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor = args->DeeployNetwork_blocks1bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_1BatchNormalization_tail_closure_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_args = (_bn_1BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks1bnweight_pert_tensor = DeeployNetwork_blocks1bnweight_pert_tensor,
      .DeeployNetwork_blocks1bnbias_pert_tensor = DeeployNetwork_blocks1bnbias_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

  // _bn_1BatchNormalization_tail_closure CLOSURE CALL
  _bn_1BatchNormalization_tail_closure(&DeeployNetwork__bn_1BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} __MERGE_CONVRQ_PASS_2_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_2_tiling_closure(void *__MERGE_CONVRQ_PASS_2_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_2_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_2_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref, 1, 16, 22,
                        14, 16, 23, 14, 16, 8, 1, 0, 0, 4, 4, 1, 1, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} __MERGE_CONVRQ_PASS_2_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_2_cluster_fork(void *__MERGE_CONVRQ_PASS_2_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_2_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_2_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_2_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_2_tiling_closure_args = (__MERGE_CONVRQ_PASS_2_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

  // __MERGE_CONVRQ_PASS_2_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_2_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} __MERGE_CONVRQ_PASS_2_closure_args_t;

static void __MERGE_CONVRQ_PASS_2_closure(void *__MERGE_CONVRQ_PASS_2_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_closure_args_t *args = (__MERGE_CONVRQ_PASS_2_closure_args_t *)__MERGE_CONVRQ_PASS_2_closure_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5152);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10080);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14240);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14176);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12128);

  const static char _MERGE_CONVRQ_PASS_2_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_2_L2_prefix[] = "[_MERGE_CONVRQ_PASS_2_L2][SB][1319878 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1446720,
          .size = 4928,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443840,
          .size = 2048,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_2_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_2_cluster_fork_args = (__MERGE_CONVRQ_PASS_2_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_2_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_2_cluster_fork_args);

    _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1315872,
          .size = 5152,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_2_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Pre-Kernel :", _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Kernel     :", _MERGE_CONVRQ_PASS_2_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Post-Kernel:", _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    uint32_t total =
        _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_2_L2_kernel_measurement + _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_2_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, _MERGE_CONVRQ_PASS_2_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} __MERGE_CONVRQ_PASS_2_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_2_closure_L3(void *__MERGE_CONVRQ_PASS_2_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_2_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_2_closure_L3_args_t *)__MERGE_CONVRQ_PASS_2_closure_L3_args;

  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_2_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_2_closure_args = (__MERGE_CONVRQ_PASS_2_closure_args_t){
      .DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = DeeployNetwork_blocks2convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_2_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_2_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_2_closure_args);

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
      14, 23, 5, 16, 1, 4, 9.999999747378752e-06f, (float32_t)0.05462646484375, (float32_t)0.0, (float32_t)23.07605633802817, (float32_t)0.0, -128, 127);

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
  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_args_t;

static void _bn_2BatchNormalization_tail_closure(void *_bn_2BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_args_t *args = (_bn_2BatchNormalization_tail_closure_args_t *)_bn_2BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor = args->DeeployNetwork_blocks2bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor = args->DeeployNetwork_blocks2bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6336);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6272);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6400);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6464);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5152);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnweight_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnbias_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_cumByteOffset
                                                                      [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_mean_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_cumByteOffset
                                                                         [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2bnrunning_var_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr]);

  const static char bn_2BatchNormalization_tail_L2_suffix[] = " cycles \n";

  const static char bn_2BatchNormalization_tail_L2_prefix[] = "[bn_2BatchNormalization_tail_L2][SB][4480 ops][Tile ";

  static uint32_t bn_2BatchNormalization_tail_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bn_2BatchNormalization_tail_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bn_2BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bn_2BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bn_2BatchNormalization_tail_L2_kernel_end_measurements[1];

  static uint32_t bn_2BatchNormalization_tail_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    bn_2BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1446944, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnweight_pert_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnbias_pert_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_mean_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_blocks2bnrunning_var_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    bn_2BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bn_2BatchNormalization_tail_L2_kernel_start_measurements[TILING_I] = getCycles();

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

    bn_2BatchNormalization_tail_L2_kernel_end_measurements[TILING_I] = getCycles();

    bn_2BatchNormalization_tail_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311840, DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    bn_2BatchNormalization_tail_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bn_2BatchNormalization_tail_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bn_2BatchNormalization_tail_L2_ingress_dma_wait_measurement = bn_2BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           bn_2BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bn_2BatchNormalization_tail_L2_kernel_measurement =
        bn_2BatchNormalization_tail_L2_kernel_end_measurements[PROFILING_I] - bn_2BatchNormalization_tail_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bn_2BatchNormalization_tail_L2_egress_dma_wait_measurement = bn_2BatchNormalization_tail_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          bn_2BatchNormalization_tail_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bn_2BatchNormalization_tail_L2_prefix, PROFILING_I, "Pre-Kernel :", bn_2BatchNormalization_tail_L2_ingress_dma_wait_measurement,
           bn_2BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_2BatchNormalization_tail_L2_prefix, PROFILING_I, "Kernel     :", bn_2BatchNormalization_tail_L2_kernel_measurement,
           bn_2BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_2BatchNormalization_tail_L2_prefix, PROFILING_I, "Post-Kernel:", bn_2BatchNormalization_tail_L2_egress_dma_wait_measurement,
           bn_2BatchNormalization_tail_L2_suffix);

    uint32_t total = bn_2BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_2BatchNormalization_tail_L2_kernel_measurement +
                     bn_2BatchNormalization_tail_L2_egress_dma_wait_measurement;
    uint32_t dma = bn_2BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_2BatchNormalization_tail_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : bn_2BatchNormalization_tail_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", bn_2BatchNormalization_tail_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, bn_2BatchNormalization_tail_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
} _bn_2BatchNormalization_tail_closure_L3_args_t;

static void _bn_2BatchNormalization_tail_closure_L3(void *_bn_2BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_2BatchNormalization_tail_closure_L3_args_t *args = (_bn_2BatchNormalization_tail_closure_L3_args_t *)_bn_2BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor = args->DeeployNetwork_blocks2bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor = args->DeeployNetwork_blocks2bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_2BatchNormalization_tail_closure_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_args = (_bn_2BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks2bnweight_pert_tensor = DeeployNetwork_blocks2bnweight_pert_tensor,
      .DeeployNetwork_blocks2bnbias_pert_tensor = DeeployNetwork_blocks2bnbias_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

  // _bn_2BatchNormalization_tail_closure CLOSURE CALL
  _bn_2BatchNormalization_tail_closure(&DeeployNetwork__bn_2BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} __MERGE_CONVRQ_PASS_3_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_3_tiling_closure(void *__MERGE_CONVRQ_PASS_3_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_3_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_3_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref, 1, 16, 5,
                        14, 16, 5, 8, 32, 1, 7, 0, 0, 0, 0, 1, 1, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} __MERGE_CONVRQ_PASS_3_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_3_cluster_fork(void *__MERGE_CONVRQ_PASS_3_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_3_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_3_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_3_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_3_tiling_closure_args = (__MERGE_CONVRQ_PASS_3_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

  // __MERGE_CONVRQ_PASS_3_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_3_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} __MERGE_CONVRQ_PASS_3_closure_args_t;

static void __MERGE_CONVRQ_PASS_3_closure(void *__MERGE_CONVRQ_PASS_3_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_closure_args_t *args = (__MERGE_CONVRQ_PASS_3_closure_args_t *)__MERGE_CONVRQ_PASS_3_closure_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6656);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7904);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7776);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5376);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);

  const static char _MERGE_CONVRQ_PASS_3_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_3_L2_prefix[] = "[_MERGE_CONVRQ_PASS_3_L2][SB][286840 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1442912,
          .size = 1120,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445376,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_3_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_3_cluster_fork_args = (__MERGE_CONVRQ_PASS_3_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_3_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_3_cluster_fork_args);

    _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312000,
          .size = 1280,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_3_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Pre-Kernel :", _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Kernel     :", _MERGE_CONVRQ_PASS_3_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Post-Kernel:", _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    uint32_t total =
        _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_3_L2_kernel_measurement + _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_3_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, _MERGE_CONVRQ_PASS_3_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} __MERGE_CONVRQ_PASS_3_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_3_closure_L3(void *__MERGE_CONVRQ_PASS_3_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_3_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_3_closure_L3_args_t *)__MERGE_CONVRQ_PASS_3_closure_L3_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_3_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_3_closure_args = (__MERGE_CONVRQ_PASS_3_closure_args_t){
      .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_3_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_3_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_3_closure_args);

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
      8, 5, 5, 32, 1, 1, 9.999999747378752e-06f, (float32_t)0.04663088917732239, (float32_t)0.0, (float32_t)41.011264080100126, (float32_t)0.0, -128, 127);

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
  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_args_t;

static void _bn_3BatchNormalization_tail_closure(void *_bn_3BatchNormalization_tail_closure_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_args_t *args = (_bn_3BatchNormalization_tail_closure_args_t *)_bn_3BatchNormalization_tail_closure_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor = args->DeeployNetwork_blocks3bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor = args->DeeployNetwork_blocks3bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2560);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2688);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2816);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2944);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnweight_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnbias_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_cumByteOffset
                                                                      [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_mean_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_cumByteOffset
                                                                         [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3bnrunning_var_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_cumByteOffset
                                                                        [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr]);

  const static char bn_3BatchNormalization_tail_L2_suffix[] = " cycles \n";

  const static char bn_3BatchNormalization_tail_L2_prefix[] = "[bn_3BatchNormalization_tail_L2][SB][1280 ops][Tile ";

  static uint32_t bn_3BatchNormalization_tail_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bn_3BatchNormalization_tail_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bn_3BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bn_3BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bn_3BatchNormalization_tail_L2_kernel_end_measurements[1];

  static uint32_t bn_3BatchNormalization_tail_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr) + 1];
       TILING_I++) {

    bn_3BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1443072, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_scale_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnweight_pert_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnbias_pert_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_mean_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_mean_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_running_var_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_blocks3bnrunning_var_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    bn_3BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bn_3BatchNormalization_tail_L2_kernel_start_measurements[TILING_I] = getCycles();

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

    bn_3BatchNormalization_tail_L2_kernel_end_measurements[TILING_I] = getCycles();

    bn_3BatchNormalization_tail_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1312000, DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    bn_3BatchNormalization_tail_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bn_3BatchNormalization_tail_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bn_3BatchNormalization_tail_L2_ingress_dma_wait_measurement = bn_3BatchNormalization_tail_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                           bn_3BatchNormalization_tail_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bn_3BatchNormalization_tail_L2_kernel_measurement =
        bn_3BatchNormalization_tail_L2_kernel_end_measurements[PROFILING_I] - bn_3BatchNormalization_tail_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bn_3BatchNormalization_tail_L2_egress_dma_wait_measurement = bn_3BatchNormalization_tail_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                          bn_3BatchNormalization_tail_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bn_3BatchNormalization_tail_L2_prefix, PROFILING_I, "Pre-Kernel :", bn_3BatchNormalization_tail_L2_ingress_dma_wait_measurement,
           bn_3BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_3BatchNormalization_tail_L2_prefix, PROFILING_I, "Kernel     :", bn_3BatchNormalization_tail_L2_kernel_measurement,
           bn_3BatchNormalization_tail_L2_suffix);

    printf("%s%u] %s%6u%s", bn_3BatchNormalization_tail_L2_prefix, PROFILING_I, "Post-Kernel:", bn_3BatchNormalization_tail_L2_egress_dma_wait_measurement,
           bn_3BatchNormalization_tail_L2_suffix);

    uint32_t total = bn_3BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_3BatchNormalization_tail_L2_kernel_measurement +
                     bn_3BatchNormalization_tail_L2_egress_dma_wait_measurement;
    uint32_t dma = bn_3BatchNormalization_tail_L2_ingress_dma_wait_measurement + bn_3BatchNormalization_tail_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : bn_3BatchNormalization_tail_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", bn_3BatchNormalization_tail_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, bn_3BatchNormalization_tail_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
} _bn_3BatchNormalization_tail_closure_L3_args_t;

static void _bn_3BatchNormalization_tail_closure_L3(void *_bn_3BatchNormalization_tail_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_3BatchNormalization_tail_closure_L3_args_t *args = (_bn_3BatchNormalization_tail_closure_L3_args_t *)_bn_3BatchNormalization_tail_closure_L3_args;

  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor = args->DeeployNetwork_blocks3bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor = args->DeeployNetwork_blocks3bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_3BatchNormalization_tail_closure_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_args = (_bn_3BatchNormalization_tail_closure_args_t){
      .DeeployNetwork_blocks3bnweight_pert_tensor = DeeployNetwork_blocks3bnweight_pert_tensor,
      .DeeployNetwork_blocks3bnbias_pert_tensor = DeeployNetwork_blocks3bnbias_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

  // _bn_3BatchNormalization_tail_closure CLOSURE CALL
  _bn_3BatchNormalization_tail_closure(&DeeployNetwork__bn_3BatchNormalization_tail_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} __MERGE_CONVRQ_PASS_4_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_4_tiling_closure(void *__MERGE_CONVRQ_PASS_4_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_4_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_4_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref, 1, 16, 5,
                        8, 32, 5, 2, 32, 1, 7, 0, 0, 0, 0, 1, 1, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} __MERGE_CONVRQ_PASS_4_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_4_cluster_fork(void *__MERGE_CONVRQ_PASS_4_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_4_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_4_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_4_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_4_tiling_closure_args = (__MERGE_CONVRQ_PASS_4_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

  // __MERGE_CONVRQ_PASS_4_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_4_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} __MERGE_CONVRQ_PASS_4_closure_args_t;

static void __MERGE_CONVRQ_PASS_4_closure(void *__MERGE_CONVRQ_PASS_4_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_closure_args_t *args = (__MERGE_CONVRQ_PASS_4_closure_args_t *)__MERGE_CONVRQ_PASS_4_closure_args;

  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10752);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12480);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12352);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 12032);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);

  const static char _MERGE_CONVRQ_PASS_4_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_4_L2_prefix[] = "[_MERGE_CONVRQ_PASS_4_L2][SB][143390 ops][Tile ";

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[1];

  static uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443072,
          .size = 1280,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1448960,
          .size = 7168,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[TILING_I] = getCycles();

    __MERGE_CONVRQ_PASS_4_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_4_cluster_fork_args = (__MERGE_CONVRQ_PASS_4_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_4_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_4_cluster_fork_args);

    _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1311040,
          .size = 320,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling _MERGE_CONVRQ_PASS_4_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       PROFILING_I++) {

    uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_measurement =
        _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[PROFILING_I];

    uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_measurement =
        _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Pre-Kernel :", _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Kernel     :", _MERGE_CONVRQ_PASS_4_L2_kernel_measurement,
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%6u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Post-Kernel:", _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_measurement,
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    uint32_t total =
        _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_4_L2_kernel_measurement + _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_measurement;
    uint32_t dma = _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_measurement + _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : _MERGE_CONVRQ_PASS_4_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, _MERGE_CONVRQ_PASS_4_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} __MERGE_CONVRQ_PASS_4_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_4_closure_L3(void *__MERGE_CONVRQ_PASS_4_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_4_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_4_closure_L3_args_t *)__MERGE_CONVRQ_PASS_4_closure_L3_args;

  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_4_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_4_closure_args = (__MERGE_CONVRQ_PASS_4_closure_args_t){
      .DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = DeeployNetwork_blocks4convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_4_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_4_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_4_closure_args);

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
      1 * 2 * 5, 32, 9.999999747378752e-06f, (float32_t)0.0173187255859375, (float32_t)0.0);

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
  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_args_t;

static void _bn_4BatchNormalization_closure(void *_bn_4BatchNormalization_closure_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_args_t *args = (_bn_4BatchNormalization_closure_args_t *)_bn_4BatchNormalization_closure_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor = args->DeeployNetwork_blocks4bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor = args->DeeployNetwork_blocks4bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1728);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1600);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1856);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1984);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2368);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_inv_std_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2496);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_tensor_ref =
      (void
           *)((char *)DeeployNetwork_blocks4bnweight_pert_tensor +
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnbias_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_mean_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_cumByteOffset
                                                                         [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4bnrunning_var_tensor + DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_cumByteOffset
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

  const static char bn_4BatchNormalization_L2_suffix[] = " cycles \n";

  const static char bn_4BatchNormalization_L2_prefix[] = "[bn_4BatchNormalization_L2][SB][2240 ops][Tile ";

  static uint32_t bn_4BatchNormalization_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bn_4BatchNormalization_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bn_4BatchNormalization_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bn_4BatchNormalization_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bn_4BatchNormalization_L2_kernel_end_measurements[1];

  static uint32_t bn_4BatchNormalization_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int saved_mean_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;
  int running_mean_ExternalToLocal = -1;
  int running_var_ExternalToLocal = -1;
  int saved_inv_std_LocalToExternal = -1;
  int bias_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int scale_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr) + 1];
       TILING_I++) {

    bn_4BatchNormalization_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1442112,
          .size = 320,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    scale_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_scale_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnweight_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    bias_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bias_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnbias_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_mean_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_mean_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_mean_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    running_var_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_running_var_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_blocks4bnrunning_var_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (bias_ExternalToLocal >= 0) {
      mchan_transfer_wait(bias_ExternalToLocal);
      mchan_transfer_free(bias_ExternalToLocal);
    }

    if (scale_ExternalToLocal >= 0) {
      mchan_transfer_wait(scale_ExternalToLocal);
      mchan_transfer_free(scale_ExternalToLocal);
    }

    if (running_mean_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_mean_ExternalToLocal);
      mchan_transfer_free(running_mean_ExternalToLocal);
    }

    if (running_var_ExternalToLocal >= 0) {
      mchan_transfer_wait(running_var_ExternalToLocal);
      mchan_transfer_free(running_var_ExternalToLocal);
    }

    bn_4BatchNormalization_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bn_4BatchNormalization_L2_kernel_start_measurements[TILING_I] = getCycles();

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

    bn_4BatchNormalization_L2_kernel_end_measurements[TILING_I] = getCycles();

    bn_4BatchNormalization_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312000,
          .size = 1280,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_data_out_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    saved_mean_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_saved_mean_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_sm_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    saved_inv_std_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
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

    bn_4BatchNormalization_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bn_4BatchNormalization_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bn_4BatchNormalization_L2_ingress_dma_wait_measurement =
        bn_4BatchNormalization_L2_ingress_dma_wait_end_measurements[PROFILING_I] - bn_4BatchNormalization_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bn_4BatchNormalization_L2_kernel_measurement =
        bn_4BatchNormalization_L2_kernel_end_measurements[PROFILING_I] - bn_4BatchNormalization_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bn_4BatchNormalization_L2_egress_dma_wait_measurement =
        bn_4BatchNormalization_L2_egress_dma_wait_end_measurements[PROFILING_I] - bn_4BatchNormalization_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_L2_prefix, PROFILING_I, "Pre-Kernel :", bn_4BatchNormalization_L2_ingress_dma_wait_measurement,
           bn_4BatchNormalization_L2_suffix);

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_L2_prefix, PROFILING_I, "Kernel     :", bn_4BatchNormalization_L2_kernel_measurement,
           bn_4BatchNormalization_L2_suffix);

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_L2_prefix, PROFILING_I, "Post-Kernel:", bn_4BatchNormalization_L2_egress_dma_wait_measurement,
           bn_4BatchNormalization_L2_suffix);

    uint32_t total = bn_4BatchNormalization_L2_ingress_dma_wait_measurement + bn_4BatchNormalization_L2_kernel_measurement +
                     bn_4BatchNormalization_L2_egress_dma_wait_measurement;
    uint32_t dma = bn_4BatchNormalization_L2_ingress_dma_wait_measurement + bn_4BatchNormalization_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : bn_4BatchNormalization_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", bn_4BatchNormalization_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, bn_4BatchNormalization_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
} _bn_4BatchNormalization_closure_L3_args_t;

static void _bn_4BatchNormalization_closure_L3(void *_bn_4BatchNormalization_closure_L3_args) {
  // CLOSURE ARG CAST
  _bn_4BatchNormalization_closure_L3_args_t *args = (_bn_4BatchNormalization_closure_L3_args_t *)_bn_4BatchNormalization_closure_L3_args;

  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor = args->DeeployNetwork_blocks4bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor = args->DeeployNetwork_blocks4bnbias_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed =
      args->DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor = args->DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor = args->DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _bn_4BatchNormalization_closure_args_t DeeployNetwork__bn_4BatchNormalization_closure_args = (_bn_4BatchNormalization_closure_args_t){
      .DeeployNetwork_blocks4bnweight_pert_tensor = DeeployNetwork_blocks4bnweight_pert_tensor,
      .DeeployNetwork_blocks4bnbias_pert_tensor = DeeployNetwork_blocks4bnbias_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
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

  // Transpose [1, 2, 5, 32] -> [1, 32, 2, 5] (Name: bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

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
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
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

  const static char bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_suffix[] = " cycles \n";

  const static char bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_prefix[] =
      "[bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2][SB][0 ops][Tile ";

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_end_measurements[1];

  static uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443072,
          .size = 1280,
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

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    _bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t
        DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args =
            (_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork,
                    &DeeployNetwork__bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_cluster_fork_args);

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312000,
          .size = 1280,
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

    bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2 =====\n");
  for (int PROFILING_I =
           ((*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr > 0)
                ? DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles[(
                      *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr - 1)]
                : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_numTiles
                         [*DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_tileIdxPtr];
       PROFILING_I++) {

    uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_measurement =
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_measurement =
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_start_measurements[PROFILING_I];

    uint32_t bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_measurement =
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
        bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Pre-Kernel :", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_measurement,
           bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Kernel     :", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_measurement,
           bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_suffix);

    printf("%s%u] %s%6u%s", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Post-Kernel:", bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_measurement,
           bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_suffix);

    uint32_t total = bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_measurement +
                     bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_measurement +
                     bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_measurement;
    uint32_t dma = bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_ingress_dma_wait_measurement +
                   bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage =
        (total == 0) ? 0 : bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n",
           bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I, total, kernel_percentage, overhead_percentage,
           bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transpose_L2_kernel_measurement, dma);
  }
  StartTimer();

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
  PULP_Relu_fp32_fp32(DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref, 320);

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
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_bn_4BatchNormalization_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_bn_4BatchNormalization_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_relu_4Relu_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_relu_4Relu_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr]);

  const static char relu_4Relu_L2_suffix[] = " cycles \n";

  const static char relu_4Relu_L2_prefix[] = "[relu_4Relu_L2][SB][320 ops][Tile ";

  static uint32_t relu_4Relu_L2_egress_dma_wait_end_measurements[1];

  static uint32_t relu_4Relu_L2_egress_dma_wait_start_measurements[1];

  static uint32_t relu_4Relu_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t relu_4Relu_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t relu_4Relu_L2_kernel_end_measurements[1];

  static uint32_t relu_4Relu_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr) + 1]; TILING_I++) {

    relu_4Relu_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_bn_4BatchNormalization_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    relu_4Relu_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    relu_4Relu_L2_kernel_start_measurements[TILING_I] = getCycles();

    _relu_4Relu_cluster_fork_args_t DeeployNetwork__relu_4Relu_cluster_fork_args =
        (_relu_4Relu_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_relu_4Relu_cluster_fork, &DeeployNetwork__relu_4Relu_cluster_fork_args);

    relu_4Relu_L2_kernel_end_measurements[TILING_I] = getCycles();

    relu_4Relu_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_relu_4Relu_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    relu_4Relu_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling relu_4Relu_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_relu_4Relu_tileIdxPtr]; PROFILING_I++) {

    uint32_t relu_4Relu_L2_ingress_dma_wait_measurement =
        relu_4Relu_L2_ingress_dma_wait_end_measurements[PROFILING_I] - relu_4Relu_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t relu_4Relu_L2_kernel_measurement = relu_4Relu_L2_kernel_end_measurements[PROFILING_I] - relu_4Relu_L2_kernel_start_measurements[PROFILING_I];

    uint32_t relu_4Relu_L2_egress_dma_wait_measurement =
        relu_4Relu_L2_egress_dma_wait_end_measurements[PROFILING_I] - relu_4Relu_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", relu_4Relu_L2_prefix, PROFILING_I, "Pre-Kernel :", relu_4Relu_L2_ingress_dma_wait_measurement, relu_4Relu_L2_suffix);

    printf("%s%u] %s%6u%s", relu_4Relu_L2_prefix, PROFILING_I, "Kernel     :", relu_4Relu_L2_kernel_measurement, relu_4Relu_L2_suffix);

    printf("%s%u] %s%6u%s", relu_4Relu_L2_prefix, PROFILING_I, "Post-Kernel:", relu_4Relu_L2_egress_dma_wait_measurement, relu_4Relu_L2_suffix);

    uint32_t total = relu_4Relu_L2_ingress_dma_wait_measurement + relu_4Relu_L2_kernel_measurement + relu_4Relu_L2_egress_dma_wait_measurement;
    uint32_t dma = relu_4Relu_L2_ingress_dma_wait_measurement + relu_4Relu_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : relu_4Relu_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", relu_4Relu_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, relu_4Relu_L2_kernel_measurement, dma);
  }
  StartTimer();

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
                              DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref, 1, 32, 2, 5);

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
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_relu_4Relu_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_relu_4Relu_output_0_tensor + DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_cumByteOffset
                                                                       [*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_global_poolGlobalAveragePool_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr]);

  const static char global_poolGlobalAveragePool_L2_suffix[] = " cycles \n";

  const static char global_poolGlobalAveragePool_L2_prefix[] = "[global_poolGlobalAveragePool_L2][SB][0 ops][Tile ";

  static uint32_t global_poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[1];

  static uint32_t global_poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[1];

  static uint32_t global_poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t global_poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t global_poolGlobalAveragePool_L2_kernel_end_measurements[1];

  static uint32_t global_poolGlobalAveragePool_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr) + 1];
       TILING_I++) {

    global_poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_relu_4Relu_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    global_poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    global_poolGlobalAveragePool_L2_kernel_start_measurements[TILING_I] = getCycles();

    _global_poolGlobalAveragePool_cluster_fork_args_t DeeployNetwork__global_poolGlobalAveragePool_cluster_fork_args =
        (_global_poolGlobalAveragePool_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_in_ref,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_global_poolGlobalAveragePool_cluster_fork, &DeeployNetwork__global_poolGlobalAveragePool_cluster_fork_args);

    global_poolGlobalAveragePool_L2_kernel_end_measurements[TILING_I] = getCycles();

    global_poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_global_poolGlobalAveragePool_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    global_poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling global_poolGlobalAveragePool_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_global_poolGlobalAveragePool_tileIdxPtr];
       PROFILING_I++) {

    uint32_t global_poolGlobalAveragePool_L2_ingress_dma_wait_measurement = global_poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                            global_poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t global_poolGlobalAveragePool_L2_kernel_measurement =
        global_poolGlobalAveragePool_L2_kernel_end_measurements[PROFILING_I] - global_poolGlobalAveragePool_L2_kernel_start_measurements[PROFILING_I];

    uint32_t global_poolGlobalAveragePool_L2_egress_dma_wait_measurement = global_poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                           global_poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", global_poolGlobalAveragePool_L2_prefix, PROFILING_I, "Pre-Kernel :", global_poolGlobalAveragePool_L2_ingress_dma_wait_measurement,
           global_poolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%6u%s", global_poolGlobalAveragePool_L2_prefix, PROFILING_I, "Kernel     :", global_poolGlobalAveragePool_L2_kernel_measurement,
           global_poolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%6u%s", global_poolGlobalAveragePool_L2_prefix, PROFILING_I, "Post-Kernel:", global_poolGlobalAveragePool_L2_egress_dma_wait_measurement,
           global_poolGlobalAveragePool_L2_suffix);

    uint32_t total = global_poolGlobalAveragePool_L2_ingress_dma_wait_measurement + global_poolGlobalAveragePool_L2_kernel_measurement +
                     global_poolGlobalAveragePool_L2_egress_dma_wait_measurement;
    uint32_t dma = global_poolGlobalAveragePool_L2_ingress_dma_wait_measurement + global_poolGlobalAveragePool_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : global_poolGlobalAveragePool_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", global_poolGlobalAveragePool_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, global_poolGlobalAveragePool_L2_kernel_measurement, dma);
  }
  StartTimer();

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
  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure(void *_wrappedInnerForwardImpl_5Gemm_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor = args->DeeployNetwork_fcbias_fp32_pert_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor = args->DeeployNetwork_fcweight_fp32_pert_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1316);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_tensor_ref =
      (void *)((char *)DeeployNetwork_Reshape_output_0_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_cumByteOffset
                                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_fcweight_fp32_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_cumByteOffset
                                                                      [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_fcbias_fp32_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_cumByteOffset
                                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_output_tensor_ref =
      (void *)((char *)DeeployNetwork_output_tensor + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_cumByteOffset
                                                          [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr]);

  const static char wrappedInnerForwardImpl_5Gemm_L2_suffix[] = " cycles \n";

  const static char wrappedInnerForwardImpl_5Gemm_L2_prefix[] = "[wrappedInnerForwardImpl_5Gemm_L2][SB][603 ops][Tile ";

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_end_measurements[1];

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_start_measurements[1];

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_kernel_end_measurements[1];

  static uint32_t wrappedInnerForwardImpl_5Gemm_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int C_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int A_ExternalToLocal = -1;
  int B_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    A_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_Reshape_output_0_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    B_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcweight_fp32_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    C_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_fcbias_fp32_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (C_ExternalToLocal >= 0) {
      mchan_transfer_wait(C_ExternalToLocal);
      mchan_transfer_free(C_ExternalToLocal);
    }

    if (A_ExternalToLocal >= 0) {
      mchan_transfer_wait(A_ExternalToLocal);
      mchan_transfer_free(A_ExternalToLocal);
    }

    if (B_ExternalToLocal >= 0) {
      mchan_transfer_wait(B_ExternalToLocal);
      mchan_transfer_free(B_ExternalToLocal);
    }

    wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    wrappedInnerForwardImpl_5Gemm_L2_kernel_start_measurements[TILING_I] = getCycles();

    _wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_cluster_fork_args =
        (_wrappedInnerForwardImpl_5Gemm_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_A_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_B_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_C_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_5Gemm_cluster_fork, &DeeployNetwork__wrappedInnerForwardImpl_5Gemm_cluster_fork_args);

    wrappedInnerForwardImpl_5Gemm_L2_kernel_end_measurements[TILING_I] = getCycles();

    wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

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

    wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling wrappedInnerForwardImpl_5Gemm_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[(
                                    *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr];
       PROFILING_I++) {

    uint32_t wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_measurement = wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
                                                                             wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t wrappedInnerForwardImpl_5Gemm_L2_kernel_measurement =
        wrappedInnerForwardImpl_5Gemm_L2_kernel_end_measurements[PROFILING_I] - wrappedInnerForwardImpl_5Gemm_L2_kernel_start_measurements[PROFILING_I];

    uint32_t wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_measurement = wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_end_measurements[PROFILING_I] -
                                                                            wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", wrappedInnerForwardImpl_5Gemm_L2_prefix, PROFILING_I, "Pre-Kernel :", wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_measurement,
           wrappedInnerForwardImpl_5Gemm_L2_suffix);

    printf("%s%u] %s%6u%s", wrappedInnerForwardImpl_5Gemm_L2_prefix, PROFILING_I, "Kernel     :", wrappedInnerForwardImpl_5Gemm_L2_kernel_measurement,
           wrappedInnerForwardImpl_5Gemm_L2_suffix);

    printf("%s%u] %s%6u%s", wrappedInnerForwardImpl_5Gemm_L2_prefix, PROFILING_I, "Post-Kernel:", wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_measurement,
           wrappedInnerForwardImpl_5Gemm_L2_suffix);

    uint32_t total = wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_measurement + wrappedInnerForwardImpl_5Gemm_L2_kernel_measurement +
                     wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_measurement;
    uint32_t dma = wrappedInnerForwardImpl_5Gemm_L2_ingress_dma_wait_measurement + wrappedInnerForwardImpl_5Gemm_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : wrappedInnerForwardImpl_5Gemm_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", wrappedInnerForwardImpl_5Gemm_L2_prefix, PROFILING_I, total,
           kernel_percentage, overhead_percentage, wrappedInnerForwardImpl_5Gemm_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
} _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t;

static void _wrappedInnerForwardImpl_5Gemm_closure_L3(void *_wrappedInnerForwardImpl_5Gemm_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *args = (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t *)_wrappedInnerForwardImpl_5Gemm_closure_L3_args;

  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor = args->DeeployNetwork_fcbias_fp32_pert_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor = args->DeeployNetwork_fcweight_fp32_pert_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor = args->DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_5Gemm_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args = (_wrappedInnerForwardImpl_5Gemm_closure_args_t){
      .DeeployNetwork_fcbias_fp32_pert_tensor = DeeployNetwork_fcbias_fp32_pert_tensor,
      .DeeployNetwork_fcweight_fp32_pert_tensor = DeeployNetwork_fcweight_fp32_pert_tensor,
      .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
      .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

  // _wrappedInnerForwardImpl_5Gemm_closure CLOSURE CALL
  _wrappedInnerForwardImpl_5Gemm_closure(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_args);

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
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
} _CrossEntropyLoss_closure_args_t;

static void _CrossEntropyLoss_closure(void *_CrossEntropyLoss_closure_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_closure_args_t *args = (_CrossEntropyLoss_closure_args_t *)_CrossEntropyLoss_closure_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 48);
  int64_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 36);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 44);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_tensor_ref =
      (void *)((char *)DeeployNetwork_output_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_1_ref =
      (void *)((char *)DeeployNetwork_output_1 +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]);

  const static char CrossEntropyLoss_L2_suffix[] = " cycles \n";

  const static char CrossEntropyLoss_L2_prefix[] = "[CrossEntropyLoss_L2][SB][0 ops][Tile ";

  static uint32_t CrossEntropyLoss_L2_egress_dma_wait_end_measurements[1];

  static uint32_t CrossEntropyLoss_L2_egress_dma_wait_start_measurements[1];

  static uint32_t CrossEntropyLoss_L2_ingress_dma_wait_end_measurements[1];

  static uint32_t CrossEntropyLoss_L2_ingress_dma_wait_start_measurements[1];

  static uint32_t CrossEntropyLoss_L2_kernel_end_measurements[1];

  static uint32_t CrossEntropyLoss_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  int logits_ExternalToLocal = -1;
  int loss_LocalToExternal = -1;
  int labels_ExternalToLocal = -1;
  int log_prob_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr) + 1]; TILING_I++) {

    CrossEntropyLoss_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

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

    if (logits_ExternalToLocal >= 0) {
      mchan_transfer_wait(logits_ExternalToLocal);
      mchan_transfer_free(logits_ExternalToLocal);
    }

    if (labels_ExternalToLocal >= 0) {
      mchan_transfer_wait(labels_ExternalToLocal);
      mchan_transfer_free(labels_ExternalToLocal);
    }

    CrossEntropyLoss_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    CrossEntropyLoss_L2_kernel_start_measurements[TILING_I] = getCycles();

    _CrossEntropyLoss_cluster_fork_args_t DeeployNetwork__CrossEntropyLoss_cluster_fork_args = (_CrossEntropyLoss_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_logits_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_labels_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_loss_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_CrossEntropyLoss_cluster_fork, &DeeployNetwork__CrossEntropyLoss_cluster_fork_args);

    CrossEntropyLoss_L2_kernel_end_measurements[TILING_I] = getCycles();

    CrossEntropyLoss_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    log_prob_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_log_prob_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_output_1_ref};
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

    CrossEntropyLoss_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  printf("===== Profiling CrossEntropyLoss_L2 =====\n");
  for (int PROFILING_I = ((*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr > 0)
                              ? DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr]; PROFILING_I++) {

    uint32_t CrossEntropyLoss_L2_ingress_dma_wait_measurement =
        CrossEntropyLoss_L2_ingress_dma_wait_end_measurements[PROFILING_I] - CrossEntropyLoss_L2_ingress_dma_wait_start_measurements[PROFILING_I];

    uint32_t CrossEntropyLoss_L2_kernel_measurement =
        CrossEntropyLoss_L2_kernel_end_measurements[PROFILING_I] - CrossEntropyLoss_L2_kernel_start_measurements[PROFILING_I];

    uint32_t CrossEntropyLoss_L2_egress_dma_wait_measurement =
        CrossEntropyLoss_L2_egress_dma_wait_end_measurements[PROFILING_I] - CrossEntropyLoss_L2_egress_dma_wait_start_measurements[PROFILING_I];

    printf("%s%u] %s%6u%s", CrossEntropyLoss_L2_prefix, PROFILING_I, "Pre-Kernel :", CrossEntropyLoss_L2_ingress_dma_wait_measurement,
           CrossEntropyLoss_L2_suffix);

    printf("%s%u] %s%6u%s", CrossEntropyLoss_L2_prefix, PROFILING_I, "Kernel     :", CrossEntropyLoss_L2_kernel_measurement, CrossEntropyLoss_L2_suffix);

    printf("%s%u] %s%6u%s", CrossEntropyLoss_L2_prefix, PROFILING_I, "Post-Kernel:", CrossEntropyLoss_L2_egress_dma_wait_measurement,
           CrossEntropyLoss_L2_suffix);

    uint32_t total =
        CrossEntropyLoss_L2_ingress_dma_wait_measurement + CrossEntropyLoss_L2_kernel_measurement + CrossEntropyLoss_L2_egress_dma_wait_measurement;
    uint32_t dma = CrossEntropyLoss_L2_ingress_dma_wait_measurement + CrossEntropyLoss_L2_egress_dma_wait_measurement;
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : CrossEntropyLoss_L2_kernel_measurement * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\n", CrossEntropyLoss_L2_prefix, PROFILING_I, total, kernel_percentage,
           overhead_percentage, CrossEntropyLoss_L2_kernel_measurement, dma);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
} _CrossEntropyLoss_closure_L3_args_t;

static void _CrossEntropyLoss_closure_L3(void *_CrossEntropyLoss_closure_L3_args) {
  // CLOSURE ARG CAST
  _CrossEntropyLoss_closure_L3_args_t *args = (_CrossEntropyLoss_closure_L3_args_t *)_CrossEntropyLoss_closure_L3_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _CrossEntropyLoss_closure_args_t DeeployNetwork__CrossEntropyLoss_closure_args = (_CrossEntropyLoss_closure_args_t){
      .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr};

  // _CrossEntropyLoss_closure CLOSURE CALL
  _CrossEntropyLoss_closure(&DeeployNetwork__CrossEntropyLoss_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks0bnbias_pert_tensor;
  float32_t *DeeployNetwork_fcbias_fp32_pert_tensor;
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int32_t *DeeployNetwork_blocks2convbias_rqsadd_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks1bnbias_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks2bnbias_pert_tensor;
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int32_t *DeeployNetwork_blocks4convbias_rqsadd_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks3bnbias_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnweight_pert_tensor;
  float32_t *DeeployNetwork_blocks4bnbias_pert_tensor;
  float32_t *DeeployNetwork_fcweight_fp32_pert_tensor;
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork_blocks2convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork_blocks4convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_rv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_sm_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_siv_tensor;
  float32_t *DeeployNetwork_bn_4BatchNormalization_output_0_tensor;
  float32_t *DeeployNetwork_relu_4Relu_output_0_tensor;
  float32_t *DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor;
  float32_t *DeeployNetwork_Reshape_output_0_tensor;
  float32_t *DeeployNetwork_output_tensor;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_1_buffer;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_2_buffer;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_3_buffer;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_4_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    _rqsp_blocks0convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks0convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks0convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks0convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr;
    DeeployNetwork_blocks0bnweight_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56592);
    _pert_blocks0bnweight_closure_L3_args_t DeeployNetwork__pert_blocks0bnweight_closure_L3_args = (_pert_blocks0bnweight_closure_L3_args_t){
        .DeeployNetwork_blocks0bnweight_pert_tensor = DeeployNetwork_blocks0bnweight_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnweight_tileIdxPtr};

    // _pert_blocks0bnweight_closure_L3 CLOSURE CALL
    _pert_blocks0bnweight_closure_L3(&DeeployNetwork__pert_blocks0bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr;
    DeeployNetwork_blocks0bnbias_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56624);
    _pert_blocks0bnbias_closure_L3_args_t DeeployNetwork__pert_blocks0bnbias_closure_L3_args = (_pert_blocks0bnbias_closure_L3_args_t){
        .DeeployNetwork_blocks0bnbias_pert_tensor = DeeployNetwork_blocks0bnbias_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks0bnbias_tileIdxPtr};

    // _pert_blocks0bnbias_closure_L3 CLOSURE CALL
    _pert_blocks0bnbias_closure_L3(&DeeployNetwork__pert_blocks0bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr;
    DeeployNetwork_fcbias_fp32_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56524);
    _pert_fcbias_fp32_closure_L3_args_t DeeployNetwork__pert_fcbias_fp32_closure_L3_args = (_pert_fcbias_fp32_closure_L3_args_t){
        .DeeployNetwork_fcbias_fp32_pert_tensor = DeeployNetwork_fcbias_fp32_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcbias_fp32_tileIdxPtr};

    // _pert_fcbias_fp32_closure_L3 CLOSURE CALL
    _pert_fcbias_fp32_closure_L3(&DeeployNetwork__pert_fcbias_fp32_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 135264);
    _rqsp_blocks1convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks1convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks1convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks1convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 135200);
    _rqsp_blocks2convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks2convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks2convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = DeeployNetwork_blocks2convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks2convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks2convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks2convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr;
    DeeployNetwork_blocks1bnweight_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 135328);
    _pert_blocks1bnweight_closure_L3_args_t DeeployNetwork__pert_blocks1bnweight_closure_L3_args = (_pert_blocks1bnweight_closure_L3_args_t){
        .DeeployNetwork_blocks1bnweight_pert_tensor = DeeployNetwork_blocks1bnweight_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnweight_tileIdxPtr};

    // _pert_blocks1bnweight_closure_L3 CLOSURE CALL
    _pert_blocks1bnweight_closure_L3(&DeeployNetwork__pert_blocks1bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr;
    DeeployNetwork_blocks1bnbias_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 136608);
    _pert_blocks1bnbias_closure_L3_args_t DeeployNetwork__pert_blocks1bnbias_closure_L3_args = (_pert_blocks1bnbias_closure_L3_args_t){
        .DeeployNetwork_blocks1bnbias_pert_tensor = DeeployNetwork_blocks1bnbias_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks1bnbias_tileIdxPtr};

    // _pert_blocks1bnbias_closure_L3 CLOSURE CALL
    _pert_blocks1bnbias_closure_L3(&DeeployNetwork__pert_blocks1bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr;
    DeeployNetwork_blocks2bnweight_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 135392);
    _pert_blocks2bnweight_closure_L3_args_t DeeployNetwork__pert_blocks2bnweight_closure_L3_args = (_pert_blocks2bnweight_closure_L3_args_t){
        .DeeployNetwork_blocks2bnweight_pert_tensor = DeeployNetwork_blocks2bnweight_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnweight_tileIdxPtr};

    // _pert_blocks2bnweight_closure_L3 CLOSURE CALL
    _pert_blocks2bnweight_closure_L3(&DeeployNetwork__pert_blocks2bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr;
    DeeployNetwork_blocks2bnbias_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 136672);
    _pert_blocks2bnbias_closure_L3_args_t DeeployNetwork__pert_blocks2bnbias_closure_L3_args = (_pert_blocks2bnbias_closure_L3_args_t){
        .DeeployNetwork_blocks2bnbias_pert_tensor = DeeployNetwork_blocks2bnbias_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks2bnbias_tileIdxPtr};

    // _pert_blocks2bnbias_closure_L3 CLOSURE CALL
    _pert_blocks2bnbias_closure_L3(&DeeployNetwork__pert_blocks2bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks0convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _rqsp_blocks0convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks0convweight_int8_closure_L3_args =
        (_rqsp_blocks0convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr};

    // _rqsp_blocks0convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks0convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks0convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56656);
    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 136864);
    _rqsp_blocks3convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks3convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks3convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks3convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 136736);
    _rqsp_blocks4convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks4convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks4convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = DeeployNetwork_blocks4convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks4convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks4convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks4convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr;
    DeeployNetwork_blocks3bnweight_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 136992);
    _pert_blocks3bnweight_closure_L3_args_t DeeployNetwork__pert_blocks3bnweight_closure_L3_args = (_pert_blocks3bnweight_closure_L3_args_t){
        .DeeployNetwork_blocks3bnweight_pert_tensor = DeeployNetwork_blocks3bnweight_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnweight_tileIdxPtr};

    // _pert_blocks3bnweight_closure_L3 CLOSURE CALL
    _pert_blocks3bnweight_closure_L3(&DeeployNetwork__pert_blocks3bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr;
    DeeployNetwork_blocks3bnbias_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 137248);
    _pert_blocks3bnbias_closure_L3_args_t DeeployNetwork__pert_blocks3bnbias_closure_L3_args = (_pert_blocks3bnbias_closure_L3_args_t){
        .DeeployNetwork_blocks3bnbias_pert_tensor = DeeployNetwork_blocks3bnbias_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks3bnbias_tileIdxPtr};

    // _pert_blocks3bnbias_closure_L3 CLOSURE CALL
    _pert_blocks3bnbias_closure_L3(&DeeployNetwork__pert_blocks3bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr;
    DeeployNetwork_blocks4bnweight_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 137120);
    _pert_blocks4bnweight_closure_L3_args_t DeeployNetwork__pert_blocks4bnweight_closure_L3_args = (_pert_blocks4bnweight_closure_L3_args_t){
        .DeeployNetwork_blocks4bnweight_pert_tensor = DeeployNetwork_blocks4bnweight_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnweight_tileIdxPtr};

    // _pert_blocks4bnweight_closure_L3 CLOSURE CALL
    _pert_blocks4bnweight_closure_L3(&DeeployNetwork__pert_blocks4bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr;
    DeeployNetwork_blocks4bnbias_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 137376);
    _pert_blocks4bnbias_closure_L3_args_t DeeployNetwork__pert_blocks4bnbias_closure_L3_args = (_pert_blocks4bnbias_closure_L3_args_t){
        .DeeployNetwork_blocks4bnbias_pert_tensor = DeeployNetwork_blocks4bnbias_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_blocks4bnbias_tileIdxPtr};

    // _pert_blocks4bnbias_closure_L3 CLOSURE CALL
    _pert_blocks4bnbias_closure_L3(&DeeployNetwork__pert_blocks4bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr;
    DeeployNetwork_fcweight_fp32_pert_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 135456);
    _pert_fcweight_fp32_closure_L3_args_t DeeployNetwork__pert_fcweight_fp32_closure_L3_args = (_pert_fcweight_fp32_closure_L3_args_t){
        .DeeployNetwork_fcweight_fp32_pert_tensor = DeeployNetwork_fcweight_fp32_pert_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_pert_fcweight_fp32_tileIdxPtr};

    // _pert_fcweight_fp32_closure_L3 CLOSURE CALL
    _pert_fcweight_fp32_closure_L3(&DeeployNetwork__pert_fcweight_fp32_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks1convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _rqsp_blocks1convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks1convweight_int8_closure_L3_args =
        (_rqsp_blocks1convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr};

    // _rqsp_blocks1convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks1convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks1convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 148256);
    __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks2convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _rqsp_blocks2convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks2convweight_int8_closure_L3_args =
        (_rqsp_blocks2convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks2convweight_int8_pert_tensor = DeeployNetwork_blocks2convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks2convweight_int8_tileIdxPtr};

    // _rqsp_blocks2convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks2convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks2convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 150304);
    __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks2convweight_int8_pert_tensor = DeeployNetwork_blocks2convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks3convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _rqsp_blocks3convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks3convweight_int8_closure_L3_args =
        (_rqsp_blocks3convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr};

    // _rqsp_blocks3convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks3convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks3convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 144672);
    __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks4convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _rqsp_blocks4convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks4convweight_int8_closure_L3_args =
        (_rqsp_blocks4convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks4convweight_int8_pert_tensor = DeeployNetwork_blocks4convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks4convweight_int8_tileIdxPtr};

    // _rqsp_blocks4convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks4convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks4convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 137504);
    __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks4convweight_int8_pert_tensor = DeeployNetwork_blocks4convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr;
    DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args =
        (_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args_t){
            .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
            .DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_QCDQ_blocks_0_conv_input_quant_1Clip_Quant_tileIdxPtr};

    // _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3 CLOSURE CALL
    _QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3(&DeeployNetwork__QCDQ_blocks_0_conv_input_quant_1Clip_Quant_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 152352);
    __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor = DeeployNetwork_blocks_0_conv_input_quant_1Clip_output_0_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 152352);
    _bnBatchNormalization_tail_closure_L3_args_t DeeployNetwork__bnBatchNormalization_tail_closure_L3_args = (_bnBatchNormalization_tail_closure_L3_args_t){
        .DeeployNetwork_blocks0bnweight_pert_tensor = DeeployNetwork_blocks0bnweight_pert_tensor,
        .DeeployNetwork_blocks0bnbias_pert_tensor = DeeployNetwork_blocks0bnbias_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks_0_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bnBatchNormalization_tail_tileIdxPtr};

    // _bnBatchNormalization_tail_closure_L3 CLOSURE CALL
    _bnBatchNormalization_tail_closure_L3(&DeeployNetwork__bnBatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    __MERGE_CONVRQ_PASS_1_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_L3_args = (__MERGE_CONVRQ_PASS_1_closure_L3_args_t){
        .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks1convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_1_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_1_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 76272);
    _bn_1BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args =
        (_bn_1BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks1bnweight_pert_tensor = DeeployNetwork_blocks1bnweight_pert_tensor,
            .DeeployNetwork_blocks1bnbias_pert_tensor = DeeployNetwork_blocks1bnbias_pert_tensor,
            .DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_1_blocks_1_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_1BatchNormalization_tail_tileIdxPtr};

    // _bn_1BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_1BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_1BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    __MERGE_CONVRQ_PASS_2_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_2_closure_L3_args = (__MERGE_CONVRQ_PASS_2_closure_L3_args_t){
        .DeeployNetwork_blocks2convbias_rqsadd_pert_tensor = DeeployNetwork_blocks2convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks2convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_2_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_2_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_2_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 61712);
    _bn_2BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args =
        (_bn_2BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks2bnweight_pert_tensor = DeeployNetwork_blocks2bnweight_pert_tensor,
            .DeeployNetwork_blocks2bnbias_pert_tensor = DeeployNetwork_blocks2bnbias_pert_tensor,
            .DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_2_blocks_2_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_2BatchNormalization_tail_tileIdxPtr};

    // _bn_2BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_2BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_2BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    __MERGE_CONVRQ_PASS_3_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_3_closure_L3_args = (__MERGE_CONVRQ_PASS_3_closure_L3_args_t){
        .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks3convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_3_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_3_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_3_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 57840);
    _bn_3BatchNormalization_tail_closure_L3_args_t DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args =
        (_bn_3BatchNormalization_tail_closure_L3_args_t){
            .DeeployNetwork_blocks3bnweight_pert_tensor = DeeployNetwork_blocks3bnweight_pert_tensor,
            .DeeployNetwork_blocks3bnbias_pert_tensor = DeeployNetwork_blocks3bnbias_pert_tensor,
            .DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_3_blocks_3_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
            .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
                DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_bn_3BatchNormalization_tail_tileIdxPtr};

    // _bn_3BatchNormalization_tail_closure_L3 CLOSURE CALL
    _bn_3BatchNormalization_tail_closure_L3(&DeeployNetwork__bn_3BatchNormalization_tail_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
        (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 59120);
    __MERGE_CONVRQ_PASS_4_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_4_closure_L3_args = (__MERGE_CONVRQ_PASS_4_closure_L3_args_t){
        .DeeployNetwork_blocks4convbias_rqsadd_pert_tensor = DeeployNetwork_blocks4convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks4convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_input_quant_1Clip_output_0_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_4_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_4_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_4_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_bn_4BatchNormalization_tileIdxPtr;
    DeeployNetwork_bn_4BatchNormalization_bn_4BatchNormalization_output_0_tensor_pre_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
    DeeployNetwork_bn_4BatchNormalization_sm_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58096);
    DeeployNetwork_bn_4BatchNormalization_siv_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 58224);
    _bn_4BatchNormalization_closure_L3_args_t DeeployNetwork__bn_4BatchNormalization_closure_L3_args = (_bn_4BatchNormalization_closure_L3_args_t){
        .DeeployNetwork_blocks4bnweight_pert_tensor = DeeployNetwork_blocks4bnweight_pert_tensor,
        .DeeployNetwork_blocks4bnbias_pert_tensor = DeeployNetwork_blocks4bnbias_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_4_blocks_4_conv_output_quant_1Clip_output_0_tensor_pre_transposed,
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
    DeeployNetwork_bn_4BatchNormalization_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 57968);
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
    DeeployNetwork_relu_4Relu_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
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
    DeeployNetwork_global_poolGlobalAveragePool_output_0_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
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

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr;
    DeeployNetwork_output_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56688);
    _wrappedInnerForwardImpl_5Gemm_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args =
        (_wrappedInnerForwardImpl_5Gemm_closure_L3_args_t){.DeeployNetwork_fcbias_fp32_pert_tensor = DeeployNetwork_fcbias_fp32_pert_tensor,
                                                           .DeeployNetwork_fcweight_fp32_pert_tensor = DeeployNetwork_fcweight_fp32_pert_tensor,
                                                           .DeeployNetwork_Reshape_output_0_tensor = DeeployNetwork_Reshape_output_0_tensor,
                                                           .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_5Gemm_tileIdxPtr};

    // _wrappedInnerForwardImpl_5Gemm_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_5Gemm_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_5Gemm_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr;
    _CrossEntropyLoss_closure_L3_args_t DeeployNetwork__CrossEntropyLoss_closure_L3_args = (_CrossEntropyLoss_closure_L3_args_t){
        .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_CrossEntropyLoss_tileIdxPtr};

    // _CrossEntropyLoss_closure_L3 CLOSURE CALL
    _CrossEntropyLoss_closure_L3(&DeeployNetwork__CrossEntropyLoss_closure_L3_args);
  }
}

void InitTrainingNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 88472);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 88472));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 162152);

  DeeployNetwork_input_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int64_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56516);
  DeeployNetwork_input_2 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56420);
  DeeployNetwork_input_3 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56452);
  DeeployNetwork_input_4 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56096);
  DeeployNetwork_input_5 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56160);
  DeeployNetwork_input_6 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56224);
  DeeployNetwork_input_7 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56288);
  DeeployNetwork_input_8 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55456);
  DeeployNetwork_input_9 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55584);
  DeeployNetwork_input_10 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55712);
  DeeployNetwork_input_11 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55840);
  DeeployNetwork_input_12 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56484);
  DeeployNetwork_input_13 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 49952);
  DeeployNetwork_input_14 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 52000);
  DeeployNetwork_input_15 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 46368);
  DeeployNetwork_input_16 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39200);
  DeeployNetwork_input_17 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56388);
  DeeployNetwork_input_18 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55968);
  DeeployNetwork_input_19 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56032);
  DeeployNetwork_input_20 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55200);
  DeeployNetwork_input_21 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 55328);
  DeeployNetwork_input_22 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 54048);
  DeeployNetwork_input_23 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56352);
  DeeployNetwork_output_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56560);
  DeeployNetwork_output_1 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 56524);
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
