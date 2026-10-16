for i in {0..3}; do
  gcloud compute instances stop "mpi-node-$i" --zone=us-central1-a &
done
wait
