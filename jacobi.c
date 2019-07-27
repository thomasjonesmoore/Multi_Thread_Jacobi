/*
   Thomas Jones-Moore
   Mason Pratz
   CS347

   ------------------
   Jacobi's Algorithm
   -----------------

   The purpose of this program to to implement Jacobi's
   algorithm on a given data set. It will run until the
   threshold is below the desired value, then print the
   amout of time it took followed by outputting the final
   iterated data set to jacobi_output.txt.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>

// DEFINITIONS
#define MATRIX_LENGTH 1024
#define THRESHOLD_MINIMUM 0.00001

// STRUCTS
typedef struct matrix_object{
	double** values;
	double largest_threshold;
}matrix_object;

typedef struct thread_args{
	matrix_object* current_matrix;
	matrix_object* next_matrix;
	int thread_id;
	int number_of_threads;
	clock_t begin;
	pthread_barrier_t barrier_1;
	pthread_barrier_t barrier_2;
}thread_args;

typedef struct barrier {
	sem_t* s;
	int count;
	sem_t* wait;
	int num_threads;
}barrier;

// VARIABLES
struct timespec begin, end;
barrier* barrier_1;
barrier* barrier_2;


// FUNCTIONS
//----------
/* splits a given line into an array of character pointers, each points to the start
of a word in the line, and it places a null pointer at the end. arg_parse() also
replaces all spaces and ':' with '\0'. */
char** arg_parse(char* line);

/* Counts the number of words in a string separated by a whitespace character.*/
int word_count(char* line);

/* Calculates each cell value in the matrix and finds the largest threshold for that iteration. Throws the threads
 * into the first barrier until they are all there. Then the very first thread created calls the swap matrices
 * function. Here, if the program is done, it prints to the file and terminates. If not, all threads go through a
 * second barrier. When all threads out of this barrier, they all call work() again */
void work(matrix_object* current_matrix, matrix_object* next_matrix, int thread_id, int number_of_threads);

/* Checks if the program is done by comparing the largest threshold to the
 * threshold minimum. If it is less than the threshold minimum, the program
 * terminates */
int is_finished(matrix_object* current_matrix);

/* Iterates through whole matrix and prints each cell value */
void print_matrix(matrix_object* matrix);

/* Points the output of print_matrix to a file called the inputted filename */
void print_to_output(char* filename, matrix_object* matrix);

/* Swaps the pointers for the matrices when a full iteration is done */
void swap_matrices(matrix_object* current_matrix, matrix_object* next_matrix);

/* Allocates space for each matrix_object and the array of values */
void alloc_matrix(matrix_object* matrix);

/* Wrapper function called once to initiate the recursive calling of work() from all threads */
void* begin_work(void* args);

/* Allocated space for the new barrier object and the 'wait' semaphore */
barrier* barrier_new(int num_threads);

/* Each thread calls this function. Uses semaphores and count increments
 * to keep track of how many threads are in the barrier and releases
 * when all threads have entered. */
void barrier_wait(barrier* bar);


/* Entry point for the program. Argv[1] contains the number of threads
 * to use for concurrency. */
int main(int argc, char* argv[]){

	clock_gettime(CLOCK_MONOTONIC, &begin);
	if (argc < 2 || argc > 2) {
		fprintf(stderr, "Usage - './jacobi [number of threads]'\n");
		exit(1);
	}
	int num_threads = (int)strtol(argv[1], NULL, 10);

  matrix_object* current_matrix = malloc(sizeof(matrix_object*));
  matrix_object* next_matrix = malloc(sizeof(matrix_object*));
	alloc_matrix(current_matrix);
	alloc_matrix(next_matrix);

  FILE* file = fopen("input.mtx", "r");
  if (file == NULL) {
	  fprintf(stderr, "Error - No umake file found.\n");
	  exit(1);
  }
  size_t  bufsize = 0;
  char*   line    = NULL;
  ssize_t linelen = getline(&line, &bufsize, file);
	int line_number = 0;

  while(-1 != linelen) {
		char** parsed = arg_parse(line);
		for (int i = 0; i < MATRIX_LENGTH; i++) {
			current_matrix->values[line_number][i] = atof(parsed[i]);
		}
		free(parsed);
		linelen = getline(&line, &bufsize, file);
		line_number++;
  }

	barrier_1 = barrier_new(num_threads);
	barrier_2 = barrier_new(num_threads);
	pthread_t threads[num_threads];
	struct thread_args ta[num_threads];

	for (int i = 0; i < num_threads; i++) {
		ta[i].current_matrix = current_matrix;
		ta[i].next_matrix = next_matrix;
		ta[i].thread_id = i;
		ta[i].number_of_threads = num_threads;
		int check = pthread_create(&threads[i], NULL, begin_work, (void*)&ta[i]);
		if (check) {
			fprintf(stderr, "Failed to create thread. Exiting.\n");
			exit(1);
		}
	}
	pthread_join(threads[0], NULL);
}

/* Only called once. Cast args as a void* and pass it to this function. 
 * Initializes the threads as structures then calls work() */
void* begin_work(void* args) {
	struct thread_args* t_args;
	t_args = (struct thread_args*) args;
	work(t_args->current_matrix, t_args->next_matrix, t_args->thread_id, t_args->number_of_threads);
}

/* Pass in current and next matrix along with the thread_id number of 
 * each thread and the total number of threads */
void work(matrix_object* current_matrix, matrix_object* next_matrix, int thread_id, int number_of_threads) {

	int row_number = thread_id;
	while (row_number < MATRIX_LENGTH) {
		for (int cell_number = 0; cell_number < MATRIX_LENGTH; cell_number++) {

			// edge cell
			if (cell_number == 0 || cell_number == MATRIX_LENGTH-1 || row_number == 0 || row_number == MATRIX_LENGTH-1) {
				next_matrix->values[row_number][cell_number] = current_matrix->values[row_number][cell_number];
			}

			else {
				double calc =(current_matrix->values[row_number-1][cell_number]
										+ current_matrix->values[row_number+1][cell_number]
										+ current_matrix->values[row_number][cell_number-1]
										+ current_matrix->values[row_number][cell_number+1]) / 4;

				next_matrix->values[row_number][cell_number] = calc;
				double threshold = fabsf(calc - current_matrix->values[row_number][cell_number]);
				if (threshold > next_matrix->largest_threshold) {
					next_matrix->largest_threshold = threshold;
				}
			}
		}
		row_number += number_of_threads;
	}

	barrier_wait(barrier_1);
	if (thread_id == 0) {
		if (is_finished(next_matrix) == 0) {
			swap_matrices(current_matrix, next_matrix);
		}
		else {
			clock_gettime(CLOCK_MONOTONIC, &end);
			double time_spent = (end.tv_sec - begin.tv_sec);
			time_spent += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
			printf("Finished in:\t%0.10f\n", time_spent);
			print_to_output("jacobi_output.mtx", current_matrix);
		}
	}
	barrier_wait(barrier_2);
	work(current_matrix, next_matrix, thread_id, number_of_threads);
}

/* Checks if program is done iterating. Pass it the updated matrix after
 * each iteration to check. */
int is_finished(matrix_object* next_matrix) {
	if (next_matrix->largest_threshold > THRESHOLD_MINIMUM) {
		return 0;
	}
	else {
		return 1;
	}
}

/* Swaps the pointers from each matrix and resets the largest threshold
 * to zero. */
void swap_matrices(matrix_object* current_matrix, matrix_object* next_matrix) {
	matrix_object temp = *next_matrix;
	*next_matrix = *current_matrix;
	*current_matrix = temp;
	current_matrix->largest_threshold = 0;
}

/* Prints the new matrix to a file called filename. Pass it the updated matrix */
void print_to_output(char* filename, matrix_object* matrix) {
	int fd;
  fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
      perror("open failed");
      exit(1);
  }
  if (dup2(fd, 1) == -1) {
      perror("dup2 failed");
      exit(1);
  }
	print_matrix(matrix);
	exit(0);
}

/* Print matrix to terminal. Call on desired matrix */
void print_matrix(matrix_object* matrix) {
	for (int i = 0; i < MATRIX_LENGTH; i++) {
		for (int j = 0; j < MATRIX_LENGTH; j++) {
			printf("%0.10f ", matrix->values[i][j]);
		}
		printf("\n");
	}
}

/* Called twice to allocate space for the matrix. Pass in un-allocated
 * matrix. */
void alloc_matrix(matrix_object* matrix) {
	matrix->largest_threshold = 0;
	matrix->values = malloc(sizeof(double*) * MATRIX_LENGTH);
	for (int i = 0; i < MATRIX_LENGTH; i++) {
		matrix->values[i] = malloc(sizeof(double) * MATRIX_LENGTH);
	}
}

/* Called twice to create the barriers used in work(). just pass in
 * total number of threads */
barrier* barrier_new(int num_threads) {
	barrier* bar;
	bar = malloc(sizeof(barrier*));
	bar->wait = malloc(sizeof(sem_t*));
	sem_init(bar->wait, 0, 0);
	sem_init(bar->s, 0, 1);
	bar->count = 0;
	bar->num_threads = num_threads;
	return bar;
}

/* Uses semaphores and a total count counter to determine
 * if all threads are in the barrier. If they are, it releases them all. */
void barrier_wait(barrier* bar) {
	sem_wait(bar->s);
	bar->count++;
	if (bar->count == bar->num_threads) {
		for (int i = 0; i < bar->num_threads; i++) {
			sem_post(bar->wait);
		}
		bar->count = 0;
	}
	sem_post(bar->s);
	sem_wait(bar->wait);
}

/* Give it a char array and it returns total number of words seperated
 * by spaces in the line until it reaches a NULL character */
int word_count(char* line) {
	char c = 1;
	int inc = 0;
	int counter = 0;
	char lastchar = 1;
	while (c != '\0') {
		c = line[inc];
		inc++;
		if (isspace(lastchar)|| lastchar == 1) {
			if (!isspace(c) && c != '\0') {
				counter++;
			}
		}
		lastchar = c;
	}
	return counter;
}

/* Parses through line which must be seperated by spaces and puts
 * each word seperated with a space in a different spot in the char**
 * array */
char** arg_parse(char* line) {
	int wcount = word_count(line);
	char** args = (char**)malloc((wcount) * sizeof(char*));
	int position = 0;
	int wordnumber = -1;
	int loop = 1;
	while (loop) {
		char currentchar = line[position];
		if (currentchar == '\0') {
			loop = 0;
		}
		else if (isspace(currentchar) || currentchar == ':') {
			line[position] = '\0';
		}
		else if (line[position-1] == '\0') {
			wordnumber++;
			args[wordnumber] = &line[position];
		}
		position++;
	}
   return args;
}
