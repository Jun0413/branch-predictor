//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"

// Definitions for 2-bit choice predictor
#define SG  0			// go for global
#define WG  1			//
#define WL  2			// go for local
#define SL  3			//

// Macros for perceptron predictor
#define PERC_ENTRIES    256
#define PERC_HIST_BITS  32
#define PERC_TTHRESHOLD 75 // Threshold to decide if training needed
                            // (int)(1.93*PERC_HIST_BITS+14)
#define PERC_WTHRESHOLD 64 // Threshold for 8-bit signed weights

// Function prototypes
void gshare_init();
uint8_t gshare_predict(uint32_t pc);
void gshare_train(uint32_t pc, uint8_t outcome);

void tournament_init();
uint8_t tournament_global_predict(uint32_t pc, int flag);
uint8_t tournament_local_predict(uint32_t pc);
uint8_t tournament_predict(uint32_t pc, int flag);
void tournament_train(uint32_t pc, uint8_t outcome, int flag);

void perceptron_init();
static inline uint32_t perceptron_pc2index(uint32_t pc);
uint8_t perceptron_predict(uint32_t pc);
void perceptron_inc_weight(int delta, uint32_t i, uint32_t j);
void perceptron_update_ghistory(uint8_t outcome);
void perceptron_train(uint32_t pc, uint8_t outcome);


const char *studentName = "Junhao Zeng; Sixuan Feng";
const char *studentID   = "A53318133; A53275766";
const char *email       = "j4zeng@eng.ucsd.edu; sfeng@eng.ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

////////////////// gshare meta /////////////////////////////
int      ghistoryLen;
uint32_t ghistory;
uint8_t* gstate;
uint32_t gmask;                   // all-one mask on low bits


////////////////// tournament meta /////////////////////////////
uint32_t  globalHistory;          // clean history
uint32_t  globalHistoryMask;
uint8_t*  globalBHT;
int       globalBHTLen;

uint32_t* localHistoryTable;
int       localHistoryTableLen;
uint32_t  localPCMask;
uint32_t  localHistoryMask;
uint8_t*  localBHT;
int       localBHTLen;

uint8_t*  chooser;


////////////////// perceptron meta /////////////////////////////
uint8_t   perc_global_history[PERC_HIST_BITS];          // LSB as the most recent outcome
int       perc_table[PERC_ENTRIES][1 + PERC_HIST_BITS]; // a table of perceptrons
int       perc_training_amount;                         // absolute perceptron output each time
uint8_t   perc_last_pred;                               // last prediction result


//------------------------------------//
//        gshare functions            //
//------------------------------------//
void gshare_init()
{
  ghistoryLen = 1 << ghistoryBits;
  gmask       = (1 << ghistoryBits) - 1;
  
  int i;
  ghistory = 0;
  gstate   = (uint8_t*) malloc(sizeof(uint8_t) * ghistoryLen);
  for (i = 0; i < ghistoryLen; ++i)  gstate[i] = WN;
}

// Xor low bits of address and history
// Return gstate at xor
//
uint8_t gshare_predict(uint32_t pc)
{
  uint32_t xor = (ghistory ^ pc) & (gmask);
  return gstate[xor] <= WN ? NOTTAKEN : TAKEN;
}

// Update gstate first then ghistory 
//
void gshare_train(uint32_t pc, uint8_t outcome)
{
  uint32_t xor   = (ghistory ^ pc) & (gmask);
  int flag       = (gstate[xor] == SN && outcome == NOTTAKEN) || (gstate[xor] == ST && outcome == TAKEN);
  if (!flag && outcome == NOTTAKEN) --gstate[xor]; // avoid using -1 or +1 for type casting
  if (!flag && outcome == TAKEN)    ++gstate[xor];

  ghistory       = (ghistory << 1) | (outcome == NOTTAKEN ? 0 : 1);
}

//------------------------------------//
//        tournament functions        //
//------------------------------------//
void tournament_init()
{

  globalHistory     = 0;
  globalHistoryMask = (1 << ghistoryBits) - 1;
  globalBHTLen      = 1 << ghistoryBits;
  globalBHT         = (uint8_t*) malloc(sizeof(uint8_t) * globalBHTLen);
  for (int i = 0; i < globalBHTLen; ++i)     globalBHT[i] = WN;

  localBHTLen            = 1 << lhistoryBits;
  localHistoryTableLen   = 1 << pcIndexBits;
  localPCMask            = (1 << pcIndexBits) - 1;
  localHistoryMask       = (1 << lhistoryBits) - 1;
  localBHT               = (uint8_t*) malloc(sizeof(uint8_t) * localBHTLen);
  for (int i = 0; i < localBHTLen; ++i)      localBHT[i] = WN;
  localHistoryTable      = (uint32_t*) malloc(sizeof(uint32_t) * localHistoryTableLen);
  for (int i = 0; i < localHistoryTableLen; ++i)  localHistoryTable[i] = 0;

  chooser = (uint8_t*) malloc(sizeof(uint8_t) * globalBHTLen);
  for (int i = 0; i < globalBHTLen; ++i)  chooser[i] = WG;

  // print_tmeta();

}

uint8_t tournament_global_predict(uint32_t pc, int flag)
{
  if (flag) {
    uint32_t xor = (globalHistory ^ pc) & (globalHistoryMask);
    return globalBHT[xor] <= WN ? NOTTAKEN : TAKEN;
  } else {
    return globalBHT[globalHistory] <= WN ? NOTTAKEN : TAKEN;
  }
}

uint8_t tournament_local_predict(uint32_t pc)
{
  uint32_t index        = pc & localPCMask;
  uint32_t localHistory = localHistoryTable[index];
  return localBHT[localHistory] <= WN ? NOTTAKEN : TAKEN;

}

uint8_t tournament_predict(uint32_t pc, int flag)
{
  uint8_t choice = chooser[globalHistory];
  return choice <= WG ? tournament_global_predict(pc, flag) : tournament_local_predict(pc);
}

void tournament_train(uint32_t pc, uint8_t outcome, int flag)
{
  uint8_t globalOutcome = tournament_global_predict(pc, flag);
  uint8_t localOutcome  = tournament_local_predict(pc);

  // train chooser
  if (globalOutcome != localOutcome) {   
    if (globalOutcome == outcome) {                   // global is correct
      if (chooser[globalHistory] != SG) {
        chooser[globalHistory]--;
      }
    } else if (localOutcome == outcome) {            // local is correct
      if (chooser[globalHistory] != SL) {
        chooser[globalHistory]++;
      }
    }
  }

  // train global predictor
  if (flag) {
    uint32_t xor = (globalHistory ^ pc) & (globalHistoryMask);
    if (TAKEN == outcome) {                     // global is correct
      if (globalBHT[xor] != ST) { ++globalBHT[xor]; }
    } else {                                            // global is wrong
      if (globalBHT[xor] != SN) { --globalBHT[xor]; }
    }
  } else {
    if (TAKEN == outcome) {                     // global is correct
      if (globalBHT[globalHistory] != ST) { ++globalBHT[globalHistory]; }
    } else {                                            // global is wrong
      if (globalBHT[globalHistory] != SN) { --globalBHT[globalHistory]; }
    }
  }
  globalHistory = ((globalHistory << 1) | (outcome == NOTTAKEN ? 0 : 1)) & globalHistoryMask;

  // train local predictor
  uint32_t localHistoryTableIndex = pc & localPCMask;
  uint32_t localBHTIndex          = localHistoryTable[localHistoryTableIndex];
  if (TAKEN == outcome) {
    if (localBHT[localBHTIndex] != ST) { localBHT[localBHTIndex]++; }
  } else {
    if (localBHT[localBHTIndex] != SN) { localBHT[localBHTIndex]--; }
  }
  localBHTIndex = ((localBHTIndex << 1) | (outcome == NOTTAKEN ? 0 : 1)) & localHistoryMask;
  localHistoryTable[localHistoryTableIndex] = localBHTIndex;

}

//------------------------------------//
//          custom functions          //
//------------------------------------//
void perceptron_init()
{
  int i, j;
  for (i = 0; i < PERC_HIST_BITS; ++i) { perc_global_history[i] = 0; }
  for (i = 0; i < PERC_ENTRIES; ++i)
    for (j = 0; j <= PERC_HIST_BITS; ++j)
      perc_table[i][j] = 1.5;

  perc_training_amount = 0;
  perc_last_pred = NOTTAKEN;
}

static inline uint32_t perceptron_pc2index(uint32_t pc)
{
  // uint32_t _ghistory = 0;
  // int _i;
  // for (_i = PERC_HIST_BITS - 1; _i >= PERC_HIST_BITS - 32; --_i)
  // {
  //   _ghistory = (perc_global_history[_i] ? 1 : 0) | _ghistory;
  //   _ghistory = _ghistory << 1;
  // }
  // return (_ghistory % PERC_ENTRIES) ^ (pc % PERC_ENTRIES);
  return pc % PERC_ENTRIES;
}

uint8_t perceptron_predict(uint32_t pc)
{
  uint32_t index = perceptron_pc2index(pc);
  int y_out = perc_table[index][0]; // bias
  
  int i;
  for (i = 1; i <= PERC_HIST_BITS; ++i)
    y_out += (perc_global_history[PERC_HIST_BITS - i] ? 1 : -1) * perc_table[index][i];
  
  perc_training_amount = (y_out >= 0 ? y_out : -y_out);
  
  perc_last_pred = (y_out >= 0 ? TAKEN : NOTTAKEN);
  return perc_last_pred;
}

void perceptron_inc_weight(int delta, uint32_t i, uint32_t j)
{
  perc_table[i][j] += delta;
  if (perc_table[i][j] >=  PERC_WTHRESHOLD) { perc_table[i][j] =  PERC_WTHRESHOLD - 1; }
  if (perc_table[i][j] <  -PERC_WTHRESHOLD) { perc_table[i][j] = -PERC_WTHRESHOLD; }
}

void perceptron_update_ghistory(uint8_t outcome)
{
  int i;
  for (i = 1; i < PERC_HIST_BITS; ++i)
    perc_global_history[i - 1] = perc_global_history[i];
  perc_global_history[PERC_HIST_BITS - 1] = outcome;
}

void perceptron_train(uint32_t pc, uint8_t outcome)
{
  uint32_t i, index = perceptron_pc2index(pc);
  int delta;
  // update weights within [-THRESHOLD, +THRESHOLD]
  if (perc_training_amount <= PERC_TTHRESHOLD || perc_last_pred != outcome)
  {
    for (i = 0; i <= PERC_HIST_BITS; ++i)
    {
      if (i == 0) { delta = (outcome == TAKEN ? 1 : -1); }
      else
      {
        if ((perc_global_history[PERC_HIST_BITS - i] == 1 && outcome == TAKEN)
        || (perc_global_history[PERC_HIST_BITS - i] == 0 && outcome == NOTTAKEN)) {
          delta = 1;
        } else { delta = -1; }
      }
      perceptron_inc_weight(delta, index, i);
    }
  }

  // update global history
  perceptron_update_ghistory(outcome);
}

void custom_init()
{
  ghistoryBits = 10;
  lhistoryBits = 9;
  pcIndexBits = 10;


  globalHistory     = 0;
  globalHistoryMask = (1 << ghistoryBits) - 1;
  globalBHTLen      = 1 << ghistoryBits;
  globalBHT         = (uint8_t*) malloc(sizeof(uint8_t) * globalBHTLen);
  for (int i = 0; i < globalBHTLen; ++i)     globalBHT[i] = WN;

  chooser = (uint8_t*) malloc(sizeof(uint8_t) * globalBHTLen);
  for (int i = 0; i < globalBHTLen; ++i)  chooser[i] = WG;

  perceptron_init();
}

uint8_t custom_predict(uint32_t pc, int flag)
{
  uint8_t choice = chooser[globalHistory];
  return choice <= WG ? tournament_global_predict(pc, flag) : perceptron_predict(pc);
}

void custom_train(uint32_t pc, uint8_t outcome, int flag)
{
  uint8_t globalOutcome = tournament_global_predict(pc, flag);
  uint8_t localOutcome  = perceptron_predict(pc);

  // train chooser
  if (globalOutcome != localOutcome) {   
    if (globalOutcome == outcome) {                  // global is correct
      if (chooser[globalHistory] != SG) {
        chooser[globalHistory]--;
      }
    } else if (localOutcome == outcome) {            // local is correct
      if (chooser[globalHistory] != SL) {
        chooser[globalHistory]++;
      }
    }
  }

  // train global predictor
  if (TAKEN == outcome) {                            // global is correct
    if (globalBHT[globalHistory] != ST) { ++globalBHT[globalHistory]; }
  } else {                                           // global is wrong
    if (globalBHT[globalHistory] != SN) { --globalBHT[globalHistory]; }
  }
  globalHistory = ((globalHistory << 1) | (outcome == NOTTAKEN ? 0 : 1)) & globalHistoryMask;

  perceptron_train(pc, outcome);
}


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void
init_predictor()
{
  switch (bpType) {
    case STATIC:
      return;
    case GSHARE:
      gshare_init();
      return;
    case TOURNAMENT:
      tournament_init();
      return;
    case CUSTOM:
      // custom_init();
      perceptron_init();
      return;
    default:
      break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint8_t
make_prediction(uint32_t pc)
{
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
      return gshare_predict(pc);
    case TOURNAMENT:
      return tournament_predict(pc, 0);
    case CUSTOM:
      // return custom_predict(pc, 0);
      return perceptron_predict(pc);
    default:
      break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//
void
train_predictor(uint32_t pc, uint8_t outcome)
{
  switch (bpType) {
    case STATIC:
      return;
    case GSHARE:
      gshare_train(pc, outcome);
      return;
    case TOURNAMENT:
      tournament_train(pc, outcome, 0);
      return;
    case CUSTOM:
      // custom_train(pc, outcome, 0);
      perceptron_train(pc, outcome);
      return;
    default:
      break;
  }
}
