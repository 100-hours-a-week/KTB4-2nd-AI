#!/bin/sh
set -eu
# 로컬 검증용 토픽만 생성한다. 운영 토픽은 클라우드가 관리한다.
for work in trip-create trip-update story-create; do
  for direction in request result; do
    topic="local.yeodam.${work}.${direction}"
    /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists \
      --topic "$topic" --partitions 2 --replication-factor 1 \
      --config retention.ms=604800000 --config cleanup.policy=delete
    /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists \
      --topic "$topic-dlt" --partitions 2 --replication-factor 1 \
      --config retention.ms=2592000000 --config cleanup.policy=delete
  done
done
