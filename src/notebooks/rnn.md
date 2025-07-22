## Plan

Steps 1. Choose Duration Size
1. For each language summate each bucket - time zone save what the minimum for each bucket it. 
2. Create a priority list from the bucket choice
3. Randomly sample and place into priority bucket to have length correct



- Loop through all languages - and respective duration to calculate best bucket size
- Doesn't need to be perfect - just approximate

For our lowest languages (calculated by num_speaker):
1. Sort df low to high
1. Randomly fill bins weighing longer audio clips more
2. Resulting shape as long as less then find_sizes will be what we train on

How Our Algorithmn Will Work
1. We look at a person and see available options
2. We then - calculate percentage of each bin full
3. We add person to each bin based off of lowest percentage

