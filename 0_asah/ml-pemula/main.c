#include <stdio.h>
#include <stdlib.h>

const char *NIM = "123230099";
const char *NAMA = "AyodyaEnhanayoan";

typedef struct
{
    int id;
    int burst_time;
    int priority;
    int remaining_time;
    int original_index;
} Process;

void print_results(const char *algorithm_name, Process proc[], int n, int quantum)
{
    int *waiting_time = (int *)malloc(sizeof(int) * n);
    if (waiting_time == NULL)
    {
        printf("Gagal alokasi memori!\n");
        return;
    }

    int total_waiting_time = 0;

    printf("========================================\n");
    if (quantum > 0)
    {
        printf("Algoritma %s (Quantum=%d)\n", algorithm_name, quantum);
    }
    else
    {
        printf("Algoritma %s\n", algorithm_name);
    }
    printf("NIM: %s, Nama: %s\n", NIM, NAMA);
    printf("----------------------------------------\n");

    if (quantum > 0)
    {
        int time = 0;
        int completed = 0;
        for (int i = 0; i < n; i++)
            proc[i].remaining_time = proc[i].burst_time;

        while (completed != n)
        {
            for (int i = 0; i < n; i++)
            {
                if (proc[i].remaining_time > 0)
                {
                    if (proc[i].remaining_time > quantum)
                    {
                        time += quantum;
                        proc[i].remaining_time -= quantum;
                    }
                    else
                    {
                        time += proc[i].remaining_time;
                        waiting_time[proc[i].original_index] = time - proc[i].burst_time;
                        proc[i].remaining_time = 0;
                        completed++;
                    }
                }
            }
        }
    }
    else
    {
        int current_time = 0;
        for (int i = 0; i < n; i++)
        {
            waiting_time[proc[i].original_index] = current_time;
            current_time += proc[i].burst_time;
        }
    }

    printf(" Proses\tWaktu Tunggu\n");
    for (int i = 0; i < n; i++)
    {
        total_waiting_time += waiting_time[i];
        printf(" P%d\t%d\n", i + 1, waiting_time[i]);
    }

    printf("\nTotal Waktu Tunggu = %d\n", total_waiting_time);
    printf("Average Waiting Time = %d/%d = %.2f\n", total_waiting_time, n, (float)total_waiting_time / n);
    printf("========================================\n\n");

    free(waiting_time);
}

int compare_sjf(const void *a, const void *b)
{
    Process *p1 = (Process *)a;
    Process *p2 = (Process *)b;
    return p1->burst_time - p2->burst_time;
}

int compare_priority(const void *a, const void *b)
{
    Process *p1 = (Process *)a;
    Process *p2 = (Process *)b;
    return p1->priority - p2->priority;
}

int main()
{
    int n = 6;
    int burst_times[] = {17, 11, 7, 5, 9, 4};
    int priorities[] = {1, 3, 2, 4, 5, 6};
    int quantum = 5;

    Process *processes = (Process *)malloc(sizeof(Process) * n);
    if (processes == NULL)
    {
        printf("Gagal alokasi memori untuk processes!\n");
        return 1;
    }

    Process *temp_proc = (Process *)malloc(sizeof(Process) * n);
    if (temp_proc == NULL)
    {
        printf("Gagal alokasi memori untuk temp_proc!\n");
        free(processes);
        return 1;
    }

    for (int i = 0; i < n; i++)
    {
        processes[i] = (Process){i + 1, burst_times[i], priorities[i], 0, i};
    }

    for (int i = 0; i < n; i++)
        temp_proc[i] = processes[i];
    print_results("FCFS (First Come First Serve)", temp_proc, n, 0);

    for (int i = 0; i < n; i++)
        temp_proc[i] = processes[i];
    qsort(temp_proc, n, sizeof(Process), compare_sjf);
    print_results("SJF (Shortest Job First)", temp_proc, n, 0);

    for (int i = 0; i < n; i++)
        temp_proc[i] = processes[i];
    print_results("Round Robin", temp_proc, n, quantum);

    for (int i = 0; i < n; i++)
        temp_proc[i] = processes[i];
    qsort(temp_proc, n, sizeof(Process), compare_priority);
    print_results("Priority Scheduling", temp_proc, n, 0);

    free(processes);
    free(temp_proc);

    return 0;
}