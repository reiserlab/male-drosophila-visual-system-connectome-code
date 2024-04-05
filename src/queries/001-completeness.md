
# Queries

This is just an arbitrary list of queries that might be useful.

example parsed from `fetch_output_completeness()`

```cypher
WITH [5746280994] as n_search_bodyId
MATCH (n:Neuron)
WHERE n.bodyId in n_search_bodyId AND n.type = 'Tm6/14'
MATCH (n)-[e:ConnectsTo]->(:Segment)
WITH n, sum(e.weight) as total_weight
MATCH (n)-[e2:ConnectsTo]->(m:Segment)
WHERE m.status in ['Traced'] OR m.statusLabel in ['Traced']
RETURN n.bodyId as bodyId, total_weight, sum(e2.weight) as traced_weight
```

Modification after 'Traced' status was remove on 2022-02-22. Without the n.type constraint, the query runs too long to return via `c.custom_query()`.
```
MATCH(n:Neuron)
WHERE (n.`ME(R)`=True OR n.`AME(R)`=True OR n.`LO(R)`=True OR n.`LOP(R)`=True)
AND n.type in ['Tm6/14']
MATCH (n)-[e:ConnectsTo]->(m:Segment)
WITH n, sum(e.weight) as total_weight
OPTIONAL MATCH (n)-[et:ConnectsTo]->(m2:Segment)
WHERE (m2.type<>'' AND m2.type IS NOT NULL) AND m2.status in ['Anchor']
RETURN n.type, n.bodyId, total_weight, sum(et.weight) as traced_weight
```

Get all aggregations inside neo4j.
```
MATCH (n:Neuron)-[e:ConnectsTo]->(m:Segment)
WHERE 
    n.type<>'' and n.type IS NOT NULL
    AND (
    n.`ME(R)`=True OR n.`AME(R)`=True 
    OR n.`LO(R)`=True OR n.`LOP(R)`=True)
    AND n.type in ['Tm6/14']
WITH n, sum(e.weight) as total_weight
MATCH (n)-[et:ConnectsTo]->(m2:Segment)
    WHERE (m2.type<>'' AND m2.type IS NOT NULL)
    OR (m2.instance<>'' AND m2.instance IS NOT NULL)
WITH n, 1.0*sum(et.weight)/total_weight as completeness,
CASE WHEN (1.0*sum(et.weight)/total_weight > 0.15) THEN 1 ELSE 0 END AS pc15,
CASE WHEN (1.0*sum(et.weight)/total_weight > 0.25) THEN 1 ELSE 0 END AS pc25,
CASE WHEN (1.0*sum(et.weight)/total_weight > 0.40) THEN 1 ELSE 0 END AS pc40,
CASE WHEN (1.0*sum(et.weight)/total_weight > 0.50) THEN 1 ELSE 0 END AS pc50
RETURN n.type as type, count(n.bodyId) as count, min(completeness) as min,
max(completeness) as max,
percentileDisc(completeness, 0.5) as median,
sum(pc15) as complete_15pct_count, sum(pc25) as complete_25pct_count,
sum(pc40) as complete_40pct_count, sum(pc50) as complete_50pct_count
```


## Get information about nodes

Get list of all properties for neurons in the optic lobe:

```cypher
MATCH (n:Neuron) 
WHERE (
    n.`ME(R)`=True OR n.`AME(R)`=True 
    OR n.`LO(R)`=True OR n.`LOP(R)`=True) 
    AND (n.type IS NOT NULL and n.type <> '') 
WITH DISTINCT keys(n) AS keys 
UNWIND keys AS keyslst WITH DISTINCT keyslst AS fieldlst
RETURN fieldlst
```
