#define N_TRAIN_STEPS 1
#define N_ACCUM_STEPS 4
#define TRAINING_DATA_SIZE 4
#define TRAINING_NUM_DATA_INPUTS 2
#define TRAINING_GRAD_BUF_START_IDX 24
#define TRAINING_NUM_GRAD_INPUTS 22
#define TRAINING_NUM_WEIGHT_INPUTS 22
#define TRAINING_LEARNING_RATE 0.001f

void *testData_mb0_buf0 = 0;
void *testData_mb0_buf1 = 0;
void *testDataRow0[2];

void *testData_mb1_buf0 = 0;
void *testData_mb1_buf1 = 0;
void *testDataRow1[2];

void *testData_mb2_buf0 = 0;
void *testData_mb2_buf1 = 0;
void *testDataRow2[2];

void *testData_mb3_buf0 = 0;
void *testData_mb3_buf1 = 0;
void *testDataRow3[2];

void **testDataVector[4] = {testDataRow0, testDataRow1, testDataRow2, testDataRow3};

void *testInitWeight_4 = 0;
void *testInitWeight_8 = 0;
void *testInitWeight_12 = 0;
void *testInitWeight_16 = 0;
void *testInitWeight_20 = 0;
void *testInitWeights[22];

#define TRAINING_TESTDATA_L3 1
void LoadTestDataL3(void) {
  testData_mb0_buf0 = cl_ram_malloc(39200);
  load_file_to_ram(testData_mb0_buf0, "testData_mb0_buf0.hex");
  testData_mb0_buf1 = cl_ram_malloc(8);
  load_file_to_ram(testData_mb0_buf1, "testData_mb0_buf1.hex");
  testDataRow0[0] = testData_mb0_buf0;
  testDataRow0[1] = testData_mb0_buf1;
  testData_mb1_buf0 = cl_ram_malloc(39200);
  load_file_to_ram(testData_mb1_buf0, "testData_mb1_buf0.hex");
  testData_mb1_buf1 = cl_ram_malloc(8);
  load_file_to_ram(testData_mb1_buf1, "testData_mb1_buf1.hex");
  testDataRow1[0] = testData_mb1_buf0;
  testDataRow1[1] = testData_mb1_buf1;
  testData_mb2_buf0 = cl_ram_malloc(39200);
  load_file_to_ram(testData_mb2_buf0, "testData_mb2_buf0.hex");
  testData_mb2_buf1 = cl_ram_malloc(8);
  load_file_to_ram(testData_mb2_buf1, "testData_mb2_buf1.hex");
  testDataRow2[0] = testData_mb2_buf0;
  testDataRow2[1] = testData_mb2_buf1;
  testData_mb3_buf0 = cl_ram_malloc(39200);
  load_file_to_ram(testData_mb3_buf0, "testData_mb3_buf0.hex");
  testData_mb3_buf1 = cl_ram_malloc(8);
  load_file_to_ram(testData_mb3_buf1, "testData_mb3_buf1.hex");
  testDataRow3[0] = testData_mb3_buf0;
  testDataRow3[1] = testData_mb3_buf1;
  testInitWeight_4 = cl_ram_malloc(8192);
  load_file_to_ram(testInitWeight_4, "testInitWeight_4.hex");
  testInitWeight_8 = cl_ram_malloc(8192);
  load_file_to_ram(testInitWeight_8, "testInitWeight_8.hex");
  testInitWeight_12 = cl_ram_malloc(14336);
  load_file_to_ram(testInitWeight_12, "testInitWeight_12.hex");
  testInitWeight_16 = cl_ram_malloc(28672);
  load_file_to_ram(testInitWeight_16, "testInitWeight_16.hex");
  testInitWeight_20 = cl_ram_malloc(1152);
  load_file_to_ram(testInitWeight_20, "testInitWeight_20.hex");
  testInitWeights[0] = NULL;
  testInitWeights[1] = NULL;
  testInitWeights[2] = NULL;
  testInitWeights[3] = NULL;
  testInitWeights[4] = testInitWeight_4;
  testInitWeights[5] = NULL;
  testInitWeights[6] = NULL;
  testInitWeights[7] = NULL;
  testInitWeights[8] = testInitWeight_8;
  testInitWeights[9] = NULL;
  testInitWeights[10] = NULL;
  testInitWeights[11] = NULL;
  testInitWeights[12] = testInitWeight_12;
  testInitWeights[13] = NULL;
  testInitWeights[14] = NULL;
  testInitWeights[15] = NULL;
  testInitWeights[16] = testInitWeight_16;
  testInitWeights[17] = NULL;
  testInitWeights[18] = NULL;
  testInitWeights[19] = NULL;
  testInitWeights[20] = testInitWeight_20;
  testInitWeights[21] = NULL;
}
