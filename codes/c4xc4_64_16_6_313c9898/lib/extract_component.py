"""Original row-space connectivity helper."""
from algebra import bits

def components(rows,n):
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for row in rows:
        support=list(bits(row))
        for q in support[1:]: parent[root(q)]=root(support[0])
    return [[i for i in range(n) if root(i)==r] for r in sorted({root(i) for i in range(n)})]

