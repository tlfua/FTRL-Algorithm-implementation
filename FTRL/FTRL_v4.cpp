#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <string>
#include <iterator>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "MurmurHash3.h"

using namespace std;

#define ALPHA 0.0674605
#define BETA 1.0
#define LANDA_1 0.0
#define LANDA_2 0.00140438
//#define W_NUM 2^20

string ORG_FILE_NAME = "../train_l50000_tl.txt";
long infile_line_num;
string TEST_FILE_NAME = "../test_tl.txt";    // fixed
string OUTPUT_FILE_NAME = "../output.csv";

int FEATURE_NUM = 39;
char FEATURE_TITLE[39][5] = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39"};
long W_NUM = pow(2, 22);

bool interaction = 0;
bool is_pruning = 0;
int p_num;

float avg_hit_rate;
float bias;

typedef struct _uData {
    long idx_cnt;
    //float x;
    float z;
    float n;
    float w;
} uData;

auto v_udata = vector<uData*>();
//uint32_t hash[1];
uint32_t seed = 42;

float p, y;

void init_v_data()
{
    cout << "0. initailize v_data\n\n";

    for (long i = 0; i < W_NUM ; i++) {
        uData* pdata = new uData;
        pdata->idx_cnt = 0;
        //pdata->x = 0.0;
        pdata->z = 0.0;
        pdata->n = 0.0;
        pdata->w = 0.0;
        v_udata.push_back(pdata);
    }
}
//------------------------------------------------------
void process_each_line(vector<string>& v_value, vector<long>& v_idx, vector<float>& v_x, bool run_algo)
{
    vector<string> v_curr_title;  // for interaction
    vector<string> v_curr_value;  // for interaction
    vector<string> v_curr_smb;    // for check
    long idx;
    auto m_idx_to_x = unordered_map<long,float>();


    for (long i = 0; i < FEATURE_NUM; i++) {

        if ( v_value[i].compare("X") != 0 ) {  // it is not missing data
            v_curr_title.push_back(FEATURE_TITLE[i]);
            v_curr_value.push_back(v_value[i]);

            string curr_smb;
            curr_smb = FEATURE_TITLE[i];
            curr_smb += ":";
            curr_smb += v_value[i];

            char s_tmp[1000] = "";
            uint32_t hash[1];

            strcpy(s_tmp, curr_smb.c_str());
            MurmurHash3_x86_32(s_tmp, curr_smb.size(), seed, hash);

            //cout << 1 % W_NUM;
            //cout << curr_smb << " " << curr_smb.size() << " " << hash << "    ";
            //cout << hash[0] % W_NUM << "  ";
            idx = hash[0] % W_NUM;
            v_idx.push_back(idx);

            if (run_algo == 1) {
                if (m_idx_to_x.count(idx) == 0) {
                    m_idx_to_x[idx] = 1.0;
                } else {
                    m_idx_to_x[idx]++;
                }
                v_x.push_back(m_idx_to_x[idx]);

                if (v_udata[idx]->idx_cnt < p_num) {
                    v_idx.pop_back();
                    v_x.pop_back();
                }
            }
        }
    }

    if (interaction == 1) {
        for (long i = 0; i < v_curr_title.size() ; i++) {
            for (long j = i+1 ; j < v_curr_title.size() ; j++) {

                string curr_smb;
                curr_smb = v_curr_title[i];
                curr_smb += "-";
                curr_smb += v_curr_title[j];
                curr_smb += ":";
                curr_smb += v_curr_value[i];
                curr_smb += "-";
                curr_smb += v_curr_value[j];

                char s_tmp[1000] = "";
                uint32_t hash[1];

                strcpy(s_tmp, curr_smb.c_str());
                MurmurHash3_x86_32(s_tmp, curr_smb.size(), seed, hash);

                idx = hash[0] % W_NUM;
                v_idx.push_back(idx);

                if (run_algo == 1) {
                    if (m_idx_to_x.count(idx) == 0) {
                        m_idx_to_x[idx] = 1.0;
                    } else {
                        m_idx_to_x[idx]++;
                    }
                    v_x.push_back(m_idx_to_x[idx]);

                    if (v_udata[idx]->idx_cnt < p_num) {
                        v_idx.pop_back();
                        v_x.pop_back();
                    }
                }
            }
        }
    }
}
//------------------------------------------------------------------
void build_table()
{
    cout << "1. build table\n";

    std::ifstream fi;
    string line;
    long cnt_line = 0;
    float sum_y = 0.0;

    fi.open( ORG_FILE_NAME );
    if ( !fi.is_open() ) {
        std::cout << "can not open file" << ORG_FILE_NAME << "\n";
    }

    while ( getline( fi, line ) ) {
        clock_t b_start = std::clock();

        cnt_line++;

        istringstream is( line );
        auto v = vector<string>( istream_iterator<string>(is), istream_iterator<string>());
        //cout << v.size() << " ";

        y = ::atof(v[0].c_str());
        sum_y += y;

        auto v_value = vector<string>();
        auto v_idx = vector<long>();
        auto v_x = vector<float>();

        for (long i = 1; i < v.size(); i++) {  // to exclude label
            v_value.push_back(v[i]);
        }

        process_each_line(v_value, v_idx, v_x, 0);

        for (long i = 1; i < v_idx.size(); i++) {
            v_udata[v_idx[i]]->idx_cnt++;
        }

        clock_t b_end = std::clock();
    }

    infile_line_num = cnt_line;
    cout << "  input file line number = " << infile_line_num << "\n";

    avg_hit_rate = (float)sum_y/ infile_line_num;
    cout << "  average hit rate = " << avg_hit_rate << "\n";

    bias = -log((1 - avg_hit_rate)/ avg_hit_rate);
    cout << "  org bias = " << bias << "\n";

    fi.close();
}
/*
//-------------------------------------------------------------------
void do_pruning()
{
    //m_ctgrVl_to_uData
    for(auto mit = m_ctgrVl_to_uData.begin(); mit != m_ctgrVl_to_uData.end(); ){
        if (mit->second->ctgrVl_cnt <= p_num){
            //cout << mit->first << "  ";

            delete mit->second;
            m_ctgrVl_to_uData.erase(mit++);
        } else {
            mit++;
        }
    }
}
*/
//-------------------------------------------------------------------
float get_p(vector<long>& v_idx)
{
    float tmp_vl = bias;
    //float rt_p;

    for (long i = 0; i < v_idx.size(); i++) {
        tmp_vl += v_udata[v_idx[i]]->w;
    }

    return 1.0/ (1.0 + exp(-tmp_vl));
}
//------------------------------------------------------------------
void update_w(float p, float y, vector<long>& v_idx, vector<float>& v_x)
{
    float g;
    float sigma;
    float sgn_z;

    bias = bias - ALPHA * (p - y);

    for (long i = 0; i < v_idx.size(); i++) {

        g = (p - y) * v_x[i];

        sigma = (sqrt(v_udata[v_idx[i]]->n + pow(g, 2)) - sqrt(v_udata[v_idx[i]]->n)) / ALPHA;
        v_udata[v_idx[i]]->z = v_udata[v_idx[i]]->z + g - sigma * v_udata[v_idx[i]]->w;
        v_udata[v_idx[i]]->n = v_udata[v_idx[i]]->n + pow(g, 2);

        if (v_udata[v_idx[i]]->z > 0)
            sgn_z = 1.0;
        else if (v_udata[v_idx[i]]->z < 0)
            sgn_z = -1.0;

        v_udata[v_idx[i]]->w = -v_udata[v_idx[i]]->z/ ((BETA + sqrt(v_udata[v_idx[i]]->n))/ ALPHA + LANDA_2);
    }
}
//----------------------------
float get_loss(float p, float y)
{
    int y_i = (int)y;

    return y == 1 ? -log(p) : -log(1.0 - p);
}
//----------------------------
void run_FTRL()
{
    cout << "2. run Follow The Regularized Leader algo\n";

    std::ifstream fi;
    string line;
    long cnt_line = 0;
    float loss_bf_u, loss_aft_u;

    fi.open( ORG_FILE_NAME );
    if ( !fi.is_open() ) {
        std::cout << "can not open file" << ORG_FILE_NAME << "\n";
    }

    while ( getline( fi, line ) ) {
        cnt_line++;

#ifdef DEBUG
        std::cout << "[" << cnt_line << "]\n";
#endif

        istringstream is( line );
        auto v = vector<string>( istream_iterator<string>(is), istream_iterator<string>());
        //cout << v.size() << " ";

        y = ::atof(v[0].c_str());
        //
        auto v_value = vector<string>();
        auto v_idx = vector<long>();
        auto v_x = vector<float>();

        for (long i = 1; i < v.size(); i++) {  // to exclude label
            v_value.push_back(v[i]);
        }

        process_each_line(v_value, v_idx, v_x, 1);
        //
        p = get_p(v_idx);

#ifdef DEBUG
        loss_bf_u = get_loss(p, y);
        cout << "loss before update w = " << loss_bf_u << "\n";
#endif

        update_w(p, y, v_idx, v_x);

#ifdef DEBUG
        p = get_p(v_idx);
        loss_aft_u = get_loss(p, y);
        cout << "loss after update w = " << loss_aft_u << "\n";
        cout << "loss difference = " << loss_aft_u - loss_bf_u << "\n";
#endif
    }

    fi.close();
}
//-------------------------------------------------------------------
void cmpt_loss_on_trainData()
{
    cout << "3. compute loss on training data\n";

    std::ifstream fi;
    string line;
    long cnt_line = 0;
    float ttl_avg_loss = 0;

    fi.open( ORG_FILE_NAME );
    if ( !fi.is_open() ) {
        std::cout << "can not open file" << ORG_FILE_NAME << "\n";
    }

    while ( getline( fi, line ) ) {
        cnt_line++;

        istringstream is( line );
        auto v = vector<string>( istream_iterator<string>(is), istream_iterator<string>());
        //cout << v.size() << " ";

        y = ::atof(v[0].c_str());
        //
        auto v_value = vector<string>();
        auto v_idx = vector<long>();
        auto v_x = vector<float>();

        for (long i = 1; i < v.size(); i++) {  // to exclude label
            v_value.push_back(v[i]);
        }

        process_each_line(v_value, v_idx, v_x, 0);
        //
        p = get_p(v_idx);

        ttl_avg_loss += get_loss(p, y);

#ifdef DEBUG
        cout << "loss = " << get_loss(p, y)
             << ", acm_loss = " << ttl_avg_loss << "\n";
#endif

    }

    ttl_avg_loss = (float)ttl_avg_loss/ infile_line_num;
    cout << "  average loss on training file = " << ttl_avg_loss << "\n";
    //
    fi.close();
}
//-------------------------------------------------------------------
void output_pred_result()
{
    cout << "4. output prediction result\n";

    std::ifstream fi;
    fi.open( TEST_FILE_NAME );
    if ( !fi.is_open() ) {
        std::cout << "can not open file " << TEST_FILE_NAME << "\n";
    }

    std::ofstream fo;
    fo.open( OUTPUT_FILE_NAME );
    if ( !fo.is_open() ) {
        std::cout << "can not open file " << OUTPUT_FILE_NAME << "\n";
    }
    fo << "Id,Predicted\n";

    string line;
    long cnt_line = 0;
    while ( getline( fi, line ) ) {
        cnt_line++;

        istringstream is( line );
        auto v = vector<string>( istream_iterator<string>(is), istream_iterator<string>());

        y = ::atof(v[0].c_str());
        //
        auto v_value = vector<string>();
        auto v_idx = vector<long>();
        auto v_x = vector<float>();

        for (long i = 0; i < v.size(); i++) {
            v_value.push_back(v[i]);
        }

        process_each_line(v_value, v_idx, v_x, 0);
        //
        fo << cnt_line + 60000000 - 1 << "," << get_p(v_idx) << "\n";
    }

    fi.close();
    fo.close();
}
//-------------------------------------------------------------------
int main(int argc, char **argv)
{
    //ORG_FILE_NAME = "../train_l50000_tl.txt";
    ORG_FILE_NAME = argv[1];
    //string OUTPUT_FILE_NAME = "../ryan_pred.csv";
    OUTPUT_FILE_NAME = argv[2];

    interaction = (bool)atoi(argv[3]);
    is_pruning = (bool)atoi(argv[4]);

    cout << "[print basic information]\n";
    cout << "  input file name: " << ORG_FILE_NAME << "\n";
    cout << "  output file name: " << OUTPUT_FILE_NAME << "\n";
    cout << "  interaction = " << interaction << "\n";
    cout << "  is pruning = " << is_pruning << "\n";
    cout << "\n";

    // pruning
    if (is_pruning == 0)
        p_num = INT_MIN;
    else if (is_pruning == 1)
        p_num = 5;

    init_v_data();

    clock_t c_start = std::clock();
    //std::cout << "[done] spent " << (c_end_3 - c_start)/ CLOCKS_PER_SEC << "sec\n\n";

    build_table();
    clock_t c_end_1 = std::clock();
    cout << "  [done] spent " << (c_end_1 - c_start)/ CLOCKS_PER_SEC << "sec\n\n";


    run_FTRL();
    clock_t c_end_2 = std::clock();
    cout << "  [done] spent " << (c_end_2 - c_end_1)/ CLOCKS_PER_SEC << "sec\n\n";
//
    cmpt_loss_on_trainData();
    clock_t c_end_3 = std::clock();
    cout << "  [done] spent " << (c_end_3 - c_end_2)/ CLOCKS_PER_SEC << "sec\n\n";
    //

//
    output_pred_result();
    clock_t c_end_4 = std::clock();
    cout << "  [done] spent " << (c_end_4 - c_end_3)/ CLOCKS_PER_SEC << "sec\n\n";
//

    return 0;
}