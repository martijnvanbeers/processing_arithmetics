FROM python:2.7-alpine as base

FROM base as builder

RUN mkdir /install
WORKDIR /install

RUN git clone https://github.com/martijnvanbeers/processing_arithmetics.git \
    && git clone https://github.com/dieuwkehupkes/keras.git

COPY models/ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10.h5 models/diagnoses/ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10_dc8.h5 processing_arithmetics/pa_demo/models

RUN pip install ./keras \
    && pip install ./processing_arithmetics

FROM base

COPY --from=builder /install /usr/local
COPY src /app
WORKDIR /app
CMD ["gunicorn", "-w 4", "pa_demo.main:app"]

