IMG=seldonio/jupyter-lab-alibi-kale:0.23

swagger-codegen:
	git clone https://github.com/swagger-api/swagger-codegen


.PHONY: swagger-codegen/swagger-v1alpha1.yml
swagger-codegen/swagger-v1alpha1.yml: swagger-codegen
	cp swagger-v1alpha1.yml swagger-codegen

swagger-codegen/gen/out/deploy: swagger-codegen swagger-codegen/swagger-v1alpha1.yml
	cd swagger-codegen && \
		./run-in-docker.sh generate -i swagger-v1alpha1.yml -l python -o /gen/out/deploy

docker-build: samples/od/cifar10 samples/cd/cifar10 swagger-codegen/gen/out/deploy
	docker build -f Dockerfile -t ${IMG} .

docker-push:
	docker push ${IMG}

samples/od/cifar10:
	mkdir -p samples/od
	gsutil cp -r gs://seldon-models/alibi-detect/od/OutlierVAE/cifar10 samples/od

samples/cd/cifar10:
	mkdir -p samples/cd
	gsutil cp -r gs://seldon-models/alibi-detect/cd/ks/cifar10 samples/cd


.PHONY: clean
clean:
	rm -rf swagger-codegen
	rm -rf samples

