from locust import HttpUser, task, between


# ============================================================
# LOCUST USER
# ============================================================

class UsuarioDeCarga(HttpUser):

    wait_time = between(0.1, 0.5)

    # ========================================================
    # STARTUP
    # ========================================================

    def on_start(self):

        self.payload = None

        response = self.client.get(
            "/sample"
        )

        if response.status_code == 200:

            body = response.json()

            self.payload = body["payload"]

            print(
                f"✅ Payload loaded "
                f"(model_version="
                f"{body['model_version']})"
            )

        else:

            print(
                "❌ Could not load payload"
            )

            print(
                response.text
            )

    # ========================================================
    # PREDICT
    # ========================================================

    @task
    def hacer_inferencia(self):

        if self.payload is None:

            return

        with self.client.post(
            "/predict",
            json=self.payload,
            catch_response=True
        ) as response:

            if response.status_code == 200:

                response.success()

            else:

                response.failure(
                    (
                        f"Status="
                        f"{response.status_code} "
                        f"{response.text}"
                    )
                )