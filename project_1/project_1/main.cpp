#include <vector>
#include <map>
#include <deque>
#include <iostream>
#include <fstream>
#include <string>
#include <fst/vector-fst.h>

/*
prefix:
a string of n words concatenated with DELIM
e.g. n = 3, DELIM = #: hello#fst#world#
word:
a string representating a single word

Author: Willy Lai
Contact: laiw@student.ethz.ch
*/

/* typedefs */
typedef std::map<std::string, double> wordCounts;
typedef std::map<std::string, std::pair<double, wordCounts> > nGramModel;

/* function signatures */
void parse_input(string filename, nGramModel& ngm, std::map<std::string, int>& statemap, std::map<std::string, int>& arcmap);
void eval(string filename);
fst::StdVectorFst write_fst(nGramModel ngm, std::map<std::string, int> statemap, std::map<std::string, int> arcmap);
void write_syms(std::map<std::string, int> arcmap);
std::string generatePrefix(std::deque<std::string>& q);
std::string getNextPrefix(std::string prefix, std::string word);
std::vector<std::string> splitString(std::string s, std::string delim);
template <typename T> void print_list(T d);
void print_map(std::map<std::string, int> mapping);
void print_ngm(nGramModel ngm);

/* constants */
const std::string START = "<s>";
const std::string END = "</s>";
const std::string DELIM = "#";
const std::string TOKEN_DELIM = " ";

/* global variables */
nGramModel model;
double noWords;
std::map<std::string, int> statemap;
int state = 0;
std::map<std::string, int> arcmap;
int arc = 0;
int N = 2; // n-gram
char* SHORT_IN = "resources/short.in";
char* FST_FILE = "binary.fst";
char* ISYMS_FILE = "isyms.txt";
char* OSYMS_FILE = "osyms.txt";
bool interactive = false;

int main() {
	using namespace std;
	if (interactive) {
		cout << "Set N:" << endl;
		int n;
		cin >> n;
		N = n;
		while(true) {
			cout << "Choose\n  1: Train\n  2: FST\n  3: Evaluate\n  4: Quit" << endl;
			int choice;
			string input;
			cin >> choice;
			switch (choice) {
				case 1:
					cout << "Enter filename" << endl;
					cin >> input;
					parse_input(input, model, statemap, arcmap);
					break;
				case 2:
					print_ngm(model);
					print_map(statemap);
					write_fst(model, statemap, arcmap);
					write_syms(arcmap);
					break;
				case 3:
					cout << "Not available yet :(" << endl;
					break;
				case 4:
					return 0;
				default:
					return 0;
			}
		}
	} else {
		parse_input("resources/a.train", model, statemap, arcmap);
		eval("resources/a.eval");
		
		print_ngm(model);
		//print_map(statemap);
		//write_fst(model, statemap, arcmap);
		//write_syms(arcmap);
	}

	return 0;
}

void parse_input(string filename, nGramModel& ngm, std::map<std::string, int>& statemap, std::map<std::string, int>& arcmap) {
	/* parses training data */
	using namespace std;
	ifstream inputFile (filename.c_str());
	if (inputFile.is_open())
	{	
		deque<string> wordQueue;
		string line;
		string key;
		string word;
		while (getline(inputFile, line)) {
			if (line.size() > 0) {
				cout << "Reading line: " << line << endl;
				vector<string> tokens = splitString(line, TOKEN_DELIM);
				vector<string>::iterator it = tokens.begin();
				vector<string>::iterator end = tokens.end();
				// append start symbol queue
				for (int i = 1; i < N; ++i) {
					wordQueue.push_back(START);
				}
				// filling up the queue with the first prefix		
				while (wordQueue.size() < N-1 && it != end) {				
					noWords++;
					wordQueue.push_back(*it);
					it++;
					print_list(wordQueue);
				}
				// pushing next symbol to the queue, generate prefix out of the n-1 last element in the queue and store that prefix with the newly pushed word in model
				// repeat until EOF
				while (it != end) {
					noWords++;
					wordQueue.push_back(*it);
					print_list(wordQueue);
					key = generatePrefix(wordQueue);
					ngm[key].second[*it]++;
					ngm[key].first++;
					if (statemap.find(key) == statemap.end()) {
						statemap[key] = state++;
					}
					if (arcmap.find(word) == arcmap.end()) {
						arcmap[*it] = arc++;
					}
					wordQueue.pop_front();
					it++;
				}
				// append end symbol and store (prefix, END) pair in model
				wordQueue.push_back(END);
				print_list(wordQueue);
				key = generatePrefix(wordQueue);
				ngm[key].second[END]++;
				ngm[key].first++;
				if (statemap.find(key) == statemap.end()) {
					statemap[key] = state++;
				}
				wordQueue.clear();
			} else {
				cout << "skip empty line..." << endl;
			}
		}
		// closing file
		inputFile.close();
	} else {
		cout << "Unable to open file\n"; 
	}
}

void eval(string filename) {
	using namespace std;
	ifstream inputFile (filename.c_str());
	ofstream outputFile;
	outputFile.open("results.txt");
	if (inputFile.is_open()) {
		deque<string> wordQueue;
		string line;
		string key;
		string word;
		while (getline(inputFile, line)) {
			if (line.size() > 0) {
				double probability = 1;
				outputFile << "Reading line: " << line << endl;
				vector<string> tokens = splitString(line, TOKEN_DELIM);
				vector<string>::iterator it = tokens.begin();
				vector<string>::iterator end = tokens.end();
				// append start symbol queue
				for (int i = 1; i < N; ++i) {
					wordQueue.push_back(START);
				}
				// filling up the queue with the first prefix		
				while (wordQueue.size() < N-1 && it != end) {				
					noWords++;
					wordQueue.push_back(*it);
					it++;
				}
				// pushing next symbol to the queue, generate prefix out of the n-1 last element in the queue and store that prefix with the newly pushed word in model
				// repeat until EOF
				while (it != end) {
					wordQueue.push_back(*it);
					key = generatePrefix(wordQueue);
					if (model[key].second.count(*it) > 0) {
						probability *= model[key].second[*it] / model[key].first;
					} else {
						probability = 0;
						break;
					}
					wordQueue.pop_front();
					it++;
				}
				wordQueue.push_back(END);
				key = generatePrefix(wordQueue);
				if (model[key].second.count(END) > 0) {
					probability *= model[key].second[END] / model[key].first;
				} else {
					probability = 0;
				}
				outputFile << "Probability of: " << probability << endl << endl;
				wordQueue.clear();
			} else {
				cout << "skip empty line..." << endl;
			}
		}
		outputFile.close();
	} else {
		cout << "Unable to open file\n"; 
	}
}

fst::StdVectorFst write_fst(nGramModel ngm, std::map<std::string, int> statemap, std::map<std::string, int> arcmap) {
	using namespace fst;
	StdVectorFst fst;
	//add start state
	int start = statemap[START];
	fst.SetStart(start);
	// add remaining states
	for (int i = 0; i < ngm.size(); ++i) {
		fst.AddState();
	}
	// add arcs (transitions)
	for (nGramModel::const_iterator it = ngm.begin(); it != ngm.end(); ++it)
	{
		wordCounts wo = it->second.second;
		for (wordCounts::const_iterator jt = wo.begin(); jt != wo.end(); ++jt) {
			// if the destination transition symbol is not END, then add arc to destination, else mark state as accepting state
			if ((jt->first).compare(END)) {
				double weight = log((jt->second) / (it->second.first));
				fst.AddArc(statemap[it->first], StdArc(arcmap[jt->first], arcmap[jt->first], weight, statemap[getNextPrefix(it->first, jt->first)]));
			} else {
				fst.SetFinal(statemap[it->first], 0);
			}
		}
	}
	// write fst file
	fst.Write(FST_FILE);
	return fst;
}

void write_syms(std::map<std::string, int> arcmap) {
	/* write arc (transition) symbol mappings */
	using namespace std;
	ofstream imap_file(ISYMS_FILE);
	ofstream omap_file(OSYMS_FILE);
	for (map<string, int>::const_iterator it = arcmap.begin(); it != arcmap.end(); ++it) {
		imap_file << it->first << " " << it->second << endl;
		omap_file << it->first << " " << it->second << endl;
	}
	imap_file.close();
	omap_file.close();
}

std::string getNextPrefix(std::string prefix, std::string word) {
	/* takes a prefix and a word and */
	using namespace std;
	string tmp = prefix.append(word).append(DELIM);
	tmp.erase(0, tmp.find_first_of(DELIM) + 1);
	return tmp;
}

std::string generatePrefix(std::deque<std::string>& q) {
	/* takes a queue and generates a prefix of the last q.size()-1 elements */
	using namespace std;
	string prefix = "";
	for (int i = 0; i < q.size()-1; ++i) {
		string current = q[i];
		prefix.append(current.append(DELIM));
	}
	return prefix;
}

std::vector<string> splitString(std::string s, std::string delim) {
	using namespace std;
	vector<string> result;
	int start = 0;
	int pos = s.find_first_of(delim, start);
	while(pos != string::npos) {
		string subs = s.substr(start, pos-start);
		result.push_back(subs);
		cout << subs << "|";
		start = pos+1;
		pos = s.find_first_of(delim, start);
	}
	string last = s.substr(start, s.size() - start+1);
	if (last.size() > 0) {
		result.push_back(last);
		cout << last;
	}
	cout << "|";
	return result;
}

/* PRINTERS */
template <typename T> void print_list(T d) {
	/* prints a iterable list */
	using namespace std;
	for (T::iterator it = d.begin(); it != d.end(); it++) {
		cout << (*it) << ' ';
	}
	cout << endl;
}

void print_map(std::map<std::string, int> mapping) {
	/* prints a prefix-to-stateID mapping */
	using namespace std;
	cout << "State Mapping:" << endl;
	for (map<string, int>::const_iterator it = mapping.begin(); it != mapping.end(); ++it) {
		cout << "Prefix: " << it->first;
		cout << "\tState: " << it->second << endl;
	}
}

void print_ngm(nGramModel ngm) {
	/* prints the N-Gram Model as a index */
	using namespace std;
	ofstream dump;
	dump.open("ngm.txt");
	dump << "N-Gram Model:" << endl;
	for (nGramModel::const_iterator it = ngm.begin(); it != ngm.end(); ++it)
	{
		dump << "[" << it->first << ": " << it->second.first << "]" << endl;
		wordCounts wo = it->second.second;
		for (wordCounts::const_iterator jt = wo.begin(); jt != wo.end(); ++jt) {
			dump << "   \\-- " << jt->first << ": " << jt->second << endl;

		}
	}
	dump.close();
}