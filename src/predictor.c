//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"

//
// TODO:Student Information
//
const char *studentName = "NAME";
const char *studentID   = "PID";
const char *email       = "EMAIL";

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
int      ghistoryLen; //TODO: currently assume to be [1, 32]
uint32_t ghistory;
uint8_t* gstate;
uint32_t gmask;      // all-one mask on low bits

////////////////// tournament meta /////////////////////////////
uint32_t globalhistory;
uint8_t* globalBHT;
uint32_t globalmask;
int      globalhistoryLen;
uint32_t *localPattern;
int      lPatternLen;
uint8_t  *localBHT;
int      localBHTLen;
uint8_t  *chooser;

void print_gmeta()
{
  printf("ghistoryLen: %d\n", ghistoryLen);
  printf("ghistory:    %d\n", ghistory);
  printf("gstate:\n");
  int i;
  for (i = 0; i < ghistoryLen; ++i) printf("%d  ", gstate[i]);
  printf("\ngmask:     %d\n", gmask);
}
////////////////////////////////////////////////////////////

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
  ghistoryLen = 1 << ghistoryBits;
  localBHTLen = 1 << lhistoryBits;
  lPatternLen = 1 << pcIndexBits;

  globalBHT = (uint8_t*) malloc(sizeof(uint8_t) * ghistoryLen);
  for (int i = 0; i < ghistoryLen; ++i)  globalBHT[i] = WN;

  localBHT = (uint8_t*) malloc(sizeof(uint8_t) * localBHTLen);
  for (int i = 0; i < localBHTLen; ++i)  localBHT[i] = WN;

  localPattern = (uint32_t*) malloc(sizeof(uint32_t) * lPatternLen);
  for (int i = 0; i < lPatternLen; ++i)  localPattern[i] = 0;

  chooser = (uint8_t*) malloc(sizeof(uint8_t) * ghistoryLen);
  for (int i = 0; i < ghistoryLen; ++i)  chooser[i] = WN;

  ghistory = 0;
  globalmask = (1 << ghistoryBits) - 1;
}

uint8_t tournament_global_predict(uint32_t pc)
{
  uint32_t globalIndex = (globalhistory^pc) & globalmask;
  return globalBHT[globalIndex] <= WN ? NOTTAKEN : TAKEN;
}

uint8_t tournament_local_predict(uint32_t pc)
{
  uint32_t localPatternIndex = pc & ((1 << pcIndexBits) - 1);
  uint32_t localBHTIndex = localPattern[localPatternIndex];
  return localBHT[localBHTIndex] <= WN ? NOTTAKEN : TAKEN;
}

uint8_t tournament_predict(uint32_t pc)
{
  uint32_t Index = (globalhistory^pc) & globalmask;
  uint8_t choice = chooser[Index];
  return choice <= WN ? tournament_global_predict(pc) : tournament_local_predict(pc);
}

void tournament_train(uint32_t pc, uint8_t outcome)
{
  // train chooser
  uint32_t globalIndex = (globalhistory^pc) & globalmask;
  uint8_t globalOutcome = tournament_global_predict(pc);
  uint8_t localOutcome = tournament_local_predict(pc);
  if (globalOutcome != localOutcome) {   
    if (globalOutcome == outcome) {                   // global is correct
      if (chooser[globalIndex] != SN) {
        chooser[globalIndex]--;
      }
    } else {                                          // local is correct
      if (chooser[globalIndex] != ST) {
        chooser[globalIndex]++;
      }
    }
  }

  // train global predictor
  if (globalOutcome == outcome) {                     // global is correct
    if (globalBHT[globalIndex] != ST) {
      globalBHT[globalIndex]++;
    }
  } else {                                            // global is wrong
    if (globalBHT[globalIndex] != SN) {
      globalBHT[globalIndex]--;
    }
  }

  globalhistory = (globalhistory << 1) | (outcome == NOTTAKEN ? 0 : 1);

  // train local predictor
  uint32_t localPatternIndex = pc & ((1 << pcIndexBits) - 1);
  uint32_t localBHTIndex = localPattern[localPatternIndex];
  if (localOutcome == outcome) {
    if (localBHT[localBHTIndex] != ST) {
      localBHT[localBHTIndex]++;
    }
  } else {
    if (localBHT[localBHTIndex] != SN) {
      localBHT[localBHTIndex]--;
    }
  }

  localPattern[localPatternIndex] = localPattern[localPatternIndex] << 1;
  localPattern[localPatternIndex] = localPattern[localPatternIndex] | (outcome == NOTTAKEN ? 0 : 1);
  localPattern[localPatternIndex] = localPattern[localPatternIndex] & ((1 << ghistoryBits) - 1);

  return;
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
  //
  //TODO: Implement prediction scheme
  //

  // Make a prediction based on the bpType
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
      return gshare_predict(pc);
    case TOURNAMENT:
      return tournament_predict(pc);
    case CUSTOM:
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
      tournament_train(pc, outcome);
      return;
    case CUSTOM:
      return;
    default:
      break;
  }
}
