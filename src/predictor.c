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
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void
init_predictor()
{
  ghistoryLen = 1;
  gmask       = 1;
  
  int i;
  for (i = 0; i < ghistoryBits; ++i)
  {
    ghistoryLen *= 2;
    gmask        = (gmask << 1) | 1;
  }

  ghistory = 0;
  gstate   = (uint8_t*) malloc(sizeof(uint8_t) * ghistoryLen);
  for (i = 0; i < ghistoryLen; ++i)  gstate[i] = WN;

  print_gmeta();
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
  gshare_train(pc, outcome);
}
