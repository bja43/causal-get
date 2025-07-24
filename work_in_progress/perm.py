
def heaps_algorithm(arr, n=None):
    if n is None:
        n = len(arr)

    if n == 1:
        print(arr)
        return

    for i in range(n):
        heaps_algorithm(arr, n - 1)

        if n % 2 == 0:
            # even: swap i and last
            arr[i], arr[n - 1] = arr[n - 1], arr[i]  
        else:
            # odd: swap first and last
            arr[0], arr[n - 1] = arr[n - 1], arr[0]


def sjt_natural(n):

    # Assume arr is [0, 1, ..., n-1]
    arr = list(range(n))

    # 0 is initially immobile
    dirs = [-1] * n
    dirs[0] = 0  

    print(arr[:])

    while True:

        # Step 1: Find largest mobile integer
        x = -1
        i = -1
        for j in range(n):
            d = dirs[arr[j]]
            k = j + d
            if d != 0 and 0 <= k < n and arr[j] > arr[k]:
                if arr[j] > x:
                    x = arr[j]
                    i = j

        # No more mobile elements
        if i == -1: break  

        d = dirs[x]
        k = i + d

        # Step 2: Swap arr[i] and arr[k]
        arr[i], arr[k] = arr[k], arr[i]

        # Step 3: Update direction of x (now at position k)
        if k == 0 or k == n - 1 or arr[k + dirs[arr[k]]] > x:
            dirs[x] = 0

        # Step 4: Reverse directions of all elements larger than x
        for j in range(n):
            if arr[j] > x:
                dirs[arr[j]] = +1 if j < k else -1

        print(arr[:])


sjt_natural(4)
