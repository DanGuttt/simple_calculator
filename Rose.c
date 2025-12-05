#include <stdlib.h>
#include <stdio.h>
#include "Rose.h"

// ----- Cas particuliers -----

// Si n1 est négatif : on remonte vers zéro
int n1Neg(int n1, int n2){
    return first(n1 + 1, n2) - 1;
}

// Si n2 est négatif : on remonte vers zéro
int n2Neg(int n1, int n2){
    return first(n1, n2 + 1) - 1;
}

// Si n1 est zéro → résultat = n2
int n1Null(int n1, int n2){
    return n2;
}

// Si n2 est zéro → résultat = n1
int n2Null(int n1, int n2){
    return n1;
}

// ----- Fonction principale -----

int first(int n1, int n2){

    if (n1 < 0)
        return n1Neg(n1, n2);

    if (n2 < 0)
        return n2Neg(n1, n2);

    if (n1 == 0)
        return n1Null(n1, n2);

    if (n2 == 0)
        return n2Null(n1, n2);

    // Définition récursive pour l’addition :
    // (n1, n2) = 1 + (n1-1, n2)
    return first(n1 - 1, n2) + 1;
}

// ----- Programme principal -----

int main(){
    int n1, n2;

    printf("Nous allons faire l'addition avec une fonction récursive.\n");
    printf("Donne ton 1er nombre : ");
    scanf("%d", &n1);

    printf("Donne ton 2eme nombre : ");
    scanf("%d", &n2);

    printf("addition : %d\n", first(n1, n2));

    return 0;
}
